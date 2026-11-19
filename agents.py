"""The four control-room agents and their prompts."""
import asyncio
import re
import time

from mahilo.agent import BaseAgent

import eventlog

SONAR, PERISCOPE, TMA, TM = "sonar_operator", "periscope_operator", "track_motion_analyst", "track_manager"


class LockedAgent(BaseAgent):
    """BaseAgent that handles one model call at a time, and logs when each call starts and ends."""
    _lock = None

    @property
    def lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _logged(self, path, steps, factory):
        asked = time.monotonic()
        async with self.lock:
            started = time.monotonic()
            token = eventlog.current_agent.set(self.name)
            eventlog.log("agent_start", agent=self.name, console=False, steps=steps,
                         wait_seconds=round(started - asked, 2), text=path)
            reply = ""
            try:
                result = await factory()
                reply = str((result or {}).get("response", "")) if isinstance(result, dict) else ""
                return result
            except Exception as exc:
                reply = f"ERROR {type(exc).__name__}: {exc}"
                raise
            finally:
                eventlog.current_agent.reset(token)
                eventlog.log("agent_done", agent=self.name, steps=steps,
                             seconds=round(time.monotonic() - started, 2), text=f"[{path}] {reply}")

    async def process_chat_message(self, message=None, websockets=[]):
        found = [int(x) for x in re.findall(r"t=(\d+)", message or "")]
        steps = "" if not found else (str(found[0]) if min(found) == max(found) else f"{min(found)}-{max(found)}")
        return await self._logged("feed", steps, lambda: super(LockedAgent, self).process_chat_message(message, websockets))

    async def process_queue_message(self, websockets=[]):
        return await self._logged("queue", "", lambda: super(LockedAgent, self).process_queue_message(websockets))


class ListenerAgent(LockedAgent):
    """An agent that can only listen and show its decision: it has no chat_with_agent tool."""

    @property
    def tools(self):
        return [t for t in super().tools if t["function"]["name"] != "chat_with_agent"]


def limit_messages(manager):
    """Stock Mahilo does not enforce can_contact and lets a model repeat itself. Drop (and log)
    a message to someone outside the sender's can_contact, and a second report by the same sender
    to the same recipient about the same ship."""
    original = manager.send_message_to_agent
    sent = set()

    async def guarded(sender, recipient, message, **kwargs):
        sender_agent = manager.agents.get(sender)
        ship = re.search(r"ship\d+", message or "", re.I)
        key = (sender, recipient, ship.group(0).lower()) if ship else None
        reason = None
        if sender_agent is not None and recipient not in (sender_agent.can_contact or []):
            reason = "recipient not in can_contact"
        elif key is not None and key in sent:
            reason = "already reported this ship"
        if reason:
            eventlog.log("message_dropped", agent=sender, sender=sender, recipient=recipient,
                         text=f"{reason}: {message}")
            return None
        if key is not None:
            sent.add(key)
        return await original(sender=sender, recipient=recipient, message=message, **kwargs)

    manager.send_message_to_agent = guarded


TEAM = (f"Team (use these agent names exactly): {SONAR}, {PERISCOPE}, {TMA}, {TM}. "
        "No human gives you instructions; a terminal only shows your replies. Use only the numbers you "
        "are given. Do exactly the job below and nothing more: do not ask anyone questions and do not "
        "send anything except the message described. When there is nothing to report, reply exactly: ACK")

SONAR_TEXT = f"""ROLE: You are the Sonar Operator in a submarine control room.
INPUT: every second you see, for each ship: ship_name, true_bearing and rel_bearing (degrees).
TASK: For each ship keep its true_bearing from the earlier seconds and work out the change since the
last second (new bearing minus old bearing). A ship is behaving unusually ONLY when the size of that
change is more than 10 degrees. A change of 10 degrees or less is normal, even if it differs from the
change before it. Ignore Ownship.
OUTPUT: the first time a ship behaves unusually, send one message to {TM} (chat_with_agent), exactly:
"SONAR t=<t> <ship> behaving unusually since t=<t>: bearing change <change> deg"
Send it once per ship. Do not message anyone else.
{TEAM}"""

TMA_TEXT = f"""ROLE: You are the Track Motion Analyst in a submarine control room.
INPUT: every second you see, for each ship and for Ownship: speed, x and y.
TASK: For each ship (not Ownship) compare this second with the last one. Notice:
 (a) did its speed change?
 (b) did its movement change? Movement = how much x and y changed since the last second
     (new minus old). It changed if either of those steps is different from the step before.
 (c) is it now closer to Ownship than before, in x or in y (the gap to Ownship's x and y is smaller)?
A ship is a THREAT when (a) or (b) happened AND (c) is true. A ship that keeps the same speed and the
same movement every second is not a threat. Do not do any other maths.
OUTPUT: the first time a ship is a threat, send one message to {TM} (chat_with_agent), exactly:
"TMA t=<t> <ship> THREAT since t=<t>: speed <old> to <new>, movement changed <yes or no>, getting closer to Ownship"
Send it once per ship. Do not message anyone else.
{TEAM}"""

PERISCOPE_TEXT = f"""ROLE: You are the Periscope Operator in a submarine control room.
INPUT: every second you see, for some ships: ship_name and the visual label.
TASK: Read the label. A destroyer is a warship and a threat to us. Fishing boats, liners and other
labels are not threats. If the label is unknown or missing, wait.
OUTPUT: the first time you see a destroyer, send one message to {TM} (chat_with_agent), exactly:
"PERISCOPE t=<t> <ship> visual=destroyer: warship, threat to us"
Send it ONCE per ship and never again, even if the label stays destroyer. Do not message anyone else.
{TEAM}"""

TM_TEXT = f"""ROLE: You are the Track Manager in a submarine control room. You only listen: you cannot talk to
other agents and you never ask anyone anything. You get no data; you only receive reports from {SONAR},
{TMA} and {PERISCOPE}.
TASK: The decision to destroy a ship is triggered by the {PERISCOPE} reporting a destroyer.
- When the {PERISCOPE} reports a destroyer, decide at once: DESTROY that ship.
- A {SONAR} or {TMA} report that arrives first is only remembered. Never decide on those alone.
- In the reasons, list ONLY reports you have actually received about that ship. If you have received
  no {SONAR} or {TMA} report about it, write "none" for that report. Never invent a report.
- Decide once per ship. If you already decided on a ship, reply exactly: ACK
OUTPUT: show the decision with contact_human, exactly:
"TM t=<t> DECISION: DESTROY <ship>. Reasons: periscope: <report>; sonar: <report or none>; tma: <report or none>"
Otherwise reply exactly: ACK
{TEAM}"""


def _agent(name, short, text, can_contact, cls=LockedAgent):
    return cls(type=name, name=name, short_description=short, description=text, can_contact=can_contact)


def create_agents():
    return [
        _agent(SONAR, "Watches bearings and reports unusual ships to the Track Manager.", SONAR_TEXT, [TM]),
        _agent(PERISCOPE, "Reads visual labels and reports destroyers to the Track Manager.", PERISCOPE_TEXT, [TM]),
        _agent(TMA, "Analyses speed and position and reports threats to the Track Manager.", TMA_TEXT, [TM]),
        _agent(TM, "Takes the final decision to destroy a ship.", TM_TEXT, [], ListenerAgent),
    ]