"""Human-in-the-loop experiment: a person takes the Track Manager's seat.   python human_app.py

Opens a page in the browser (http://127.0.0.1:8000). The experimenter starts a session and presses Next to
release each data step. The participant can chat with the Sonar Operator, the Track Analyst and the Periscope Operator (each sees only
its own data) and submits ship type, distance, confidence and threat answers per step. Everything is logged with time stamps in
logs/<participant>_<date_time>/ .

Set OPENAI_API_KEY and MAHILO_LLM_MODEL (for example gpt-6-luna) in the terminal first. Never put the key in a file.
"""
import asyncio
import csv
import json
import os
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict

from mahilo import AgentManager

import eventlog
from agents import SONAR, limit_messages
from agents_v2 import TA
from feeder_v2 import load_steps, periscope_block, sonar_block, ta_block
from human_agents import LABELS, PERI, create_advisers, prompt_version, reference_text
from run_server_v2 import pin_model

BASE = Path(__file__).parent
FEED = os.getenv("PARTICIPANT_FEED", "scenario_feed.csv")
LOG_ROOT = Path(os.getenv("LOG_ROOT", "logs"))
KINDS = ["fishing vessel", "ferry", "cargo ship", "hostile ship"]
THREAT_LEVELS = ["none", "low", "high"]          # edit this list when the threat scale is agreed
ALLOWED_COLUMNS = {"time_step", "time_min", "contact", "own_x", "own_y", "track_x", "track_y", "range", "speed",
                   "sonar_cue", "periscope_raised", "periscope_report", "collision_warning"}
DISTANCE_MAX = 20                                  # the Distance answer is a whole number of cells, 0 to 20
CONFIDENCE = [(1, "1 - guess"), (2, "2"), (3, "3"), (4, "4"), (5, "5 - certain")]   # one rating per track
ANSWER_COLUMNS = ["submitted_wall", "elapsed", "step", "time_min", "contact", "type", "distance", "confidence",
                  "threat"]
MAX_MESSAGE = 2000


# ---------------------------------------------------------------- the data the participant may see
def load_feed():
    if not Path(FEED).exists():
        sys.exit(f"{FEED} not found. Put scenario_feed.csv in this folder.")
    with open(FEED, newline="", encoding="utf-8-sig") as handle:
        header = set(csv.DictReader(handle).fieldnames or [])
    missing = ALLOWED_COLUMNS - header
    if missing:
        sys.exit(f"{FEED} is missing columns {sorted(missing)}. Use the final scenario_feed.csv.")
    extra = header - ALLOWED_COLUMNS
    if extra:
        sys.exit(f"{FEED} has columns the participant must not see: {sorted(extra)}. Use the final scenario_feed.csv.")
    steps = load_steps(FEED)
    contacts = sorted({r["contact"] for rows in steps.values() for r in rows})
    return steps, sorted(steps), contacts


STEPS, ORDER, CONTACTS = load_feed()
def track_name(contact):
    """The participant and the agents say Track_1; files and scoring keep the internal name Contact_1."""
    return contact.replace("Contact_", "Track_")


BLOCK = {SONAR: lambda step, rows: track_name(sonar_block(step, rows)),
         TA: lambda step, rows: track_name(ta_block(step, rows)),
         PERI: lambda step, rows: track_name(periscope_block(step, rows))}
ADVISERS = {}
app = FastAPI()
_lock = None


def lock():
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def fresh_state():
    return {"active": False, "participant": "", "session_id": "", "folder": None, "next_index": 0,
            "current": None, "answered": False, "released": [], "chats": {n: [] for n in LABELS},
            "last_answers": {}, "blocks": {n: [] for n in LABELS}, "delivery": {}}


S = fresh_state()


def now():
    return datetime.now().isoformat(timespec="seconds")


def step_view(step):
    """All the page gets about a released step: its number and time. The data itself reaches the participant
    only through the agents' answers."""
    return {"step": step, "time_min": STEPS[step][0]["time_min"], "wall": now()}


def flags():
    last = S["next_index"] >= len(ORDER)
    return {"active": S["active"], "participant": S["participant"], "current": S["current"],
            "answered": S["answered"], "total": len(ORDER), "released_count": S["next_index"],
            "last_step_released": last,
            "can_next": S["active"] and not last and (S["current"] is None or S["answered"])}


# ---------------------------------------------------------------- pages and read-only calls
@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/config")
def config():
    return {"contacts": CONTACTS, "tracks": {c: track_name(c) for c in CONTACTS}, "kinds": KINDS,
            "threats": THREAT_LEVELS, "confidence": [{"value": v, "label": l} for v, l in CONFIDENCE],
            "total": len(ORDER), "distance_max": DISTANCE_MAX,
            "agents": [{"name": n, "label": LABELS[n]} for n in LABELS], "reference": reference_text()}


@app.get("/api/state")
def state():
    return {**flags(), "released": S["released"], "chats": S["chats"], "last_answers": S["last_answers"],
            "delivery": S["delivery"]}


# ---------------------------------------------------------------- experimenter controls
class StartIn(BaseModel):
    participant_id: str


@app.post("/api/start")
async def start(body: StartIn):
    pid = body.participant_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", pid):
        raise HTTPException(400, "Participant ID: letters, digits, - and _ only (max 40).")
    async with lock():
        if S["active"]:
            raise HTTPException(409, "A session is already running.")
        session_id = f"{pid}_{datetime.now():%Y%m%d_%H%M%S}"
        folder = LOG_ROOT / session_id
        eventlog.open_session(folder, pid)
        eventlog.start_clock()
        with open(folder / "answers.csv", "w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerow(ANSWER_COLUMNS)
        for agent in ADVISERS.values():
            agent.activate(server_id=session_id)             # fresh, empty history in sessions/<session_id>/
        S.clear()
        S.update(fresh_state())
        S.update(active=True, participant=pid, session_id=session_id, folder=str(folder))
        meta = {"participant": pid, "session_id": session_id, "started": now(),
                "model": os.getenv("MAHILO_LLM_MODEL", "(Mahilo default)"), "prompt_version": prompt_version(),
                "feed": FEED, "steps": len(ORDER), "threat_levels": THREAT_LEVELS,
                "confidence_scale": [v for v, _ in CONFIDENCE], "distance_max": DISTANCE_MAX}
        (folder / "session_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        eventlog.log("session_start", text=json.dumps(meta))
        return flags()


class NextIn(BaseModel):
    skip: bool = False


@app.post("/api/next")
async def next_step(body: NextIn):
    async with lock():
        if not S["active"]:
            raise HTTPException(409, "No session is running.")
        if S["next_index"] >= len(ORDER):
            raise HTTPException(409, "There are no more steps. End the session.")
        if S["current"] is not None and not S["answered"]:
            if not body.skip:
                raise HTTPException(409, "Submit the answers for the current step first.")
            eventlog.log("step_skipped", text=f"step {S['current']} left without answers")
        step = ORDER[S["next_index"]]
        eventlog.set_step(step)
        eventlog.log("next_pressed", text=f"experimenter released step {step}")
        delivery = {}
        for name, agent in ADVISERS.items():
            block = BLOCK[name](step, STEPS[step])
            async with agent.lock:                             # wait for a running answer to finish
                agent.add_data(block)
                ok = agent.holds(block)
                if not ok:                                     # one more try, then report it loudly
                    agent.add_data(block)
                    ok = agent.holds(block)
            S["blocks"][name].append(block)
            delivery[name] = ok
            eventlog.log("feed_release", agent=name, console=False, text=block)
            eventlog.log("delivery_check", agent=name, text=("ok" if ok else "NOT DELIVERED") + f": step {step}")
        S["delivery"] = {"step": step, "agents": delivery, "all_ok": all(delivery.values())}
        view = step_view(step)
        S["released"].append(view)
        S["next_index"] += 1
        S["current"], S["answered"] = step, False
        eventlog.log("feed_release", text=f"step {step} released")
        return {**flags(), "step": view, "delivery": S["delivery"]}


class AnswersIn(BaseModel):
    step: int
    answers: Dict[str, Any]


@app.post("/api/answers")
async def answers(body: AnswersIn):
    async with lock():
        if not S["active"] or S["current"] is None:
            raise HTTPException(409, "No step is waiting for answers.")
        if body.step != S["current"]:
            raise HTTPException(409, f"Answers are for step {S['current']}, not {body.step}.")
        if set(body.answers) != set(CONTACTS):
            raise HTTPException(400, "An answer is needed for every contact.")
        clean = {}
        for contact, a in body.answers.items():
            a = a if isinstance(a, dict) else {}
            kind, threat = a.get("type"), a.get("threat")
            distance, confidence = a.get("distance"), a.get("confidence")
            ok = kind in KINDS + ["unknown"] and threat in THREAT_LEVELS
            ok = ok and isinstance(distance, int) and not isinstance(distance, bool) and 0 <= distance <= DISTANCE_MAX
            ok = ok and confidence in [v for v, _ in CONFIDENCE] and not isinstance(confidence, bool)
            if not ok:
                raise HTTPException(400, f"Incomplete or invalid answer for {track_name(contact)}.")
            clean[contact] = {"type": kind, "distance": distance, "confidence": confidence, "threat": threat}
        time_min = STEPS[body.step][0]["time_min"]
        with open(Path(S["folder"]) / "answers.csv", "a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            for contact in CONTACTS:
                c = clean[contact]
                writer.writerow([now(), round(eventlog.elapsed(), 3), body.step, time_min, contact, c["type"],
                                 c["distance"], c["confidence"], c["threat"]])
        eventlog.log("answer_submit", text=json.dumps({"step": body.step, "answers": clean}))
        S["answered"], S["last_answers"] = True, clean
        return {**flags(), "saved": now()}


@app.post("/api/end")
async def end():
    async with lock():
        if not S["active"]:
            raise HTTPException(409, "No session is running.")
        early = S["next_index"] < len(ORDER) or not S["answered"]
        folder = S["folder"]
        eventlog.log("session_end", text=f"released {S['next_index']} of {len(ORDER)} steps"
                                          + ("; ended early or last step unanswered" if early else ""))
        S.clear()
        S.update(fresh_state())
        return {"folder": folder, "ended_early": early}


# ---------------------------------------------------------------- the participant's chat with the advisers
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def unverified_numbers(reply, blocks):
    """Numbers in an agent's reply that appear nowhere in the data blocks that agent received."""
    known = {float(n) for block in blocks for n in NUMBER.findall(block)}
    return sorted({n for n in NUMBER.findall(reply) if float(n) not in known}, key=float)


class ShowIn(BaseModel):
    agent: str


@app.post("/api/show")
async def show_data(body: ShowIn):
    """Show an agent's latest data exactly as it received it. No model is involved, so nothing can be left out."""
    if not S["active"]:
        raise HTTPException(409, "No session is running.")
    if body.agent not in ADVISERS:
        raise HTTPException(400, "Unknown agent.")
    blocks = S["blocks"][body.agent]
    if not blocks:
        raise HTTPException(409, "No data has been released yet.")
    text = blocks[-1]
    eventlog.log("show_data", agent=body.agent, sender=body.agent, recipient="participant", text=text)
    S["chats"][body.agent].append({"who": "you", "text": "(Show latest data)", "wall": now(), "agent": body.agent})
    S["chats"][body.agent].append({"who": "agent", "text": text, "wall": now(), "agent": body.agent})
    return {"ok": True, "text": text}


class ChatIn(BaseModel):
    agent: str
    message: str


@app.post("/api/chat")
async def chat(body: ChatIn):
    agent = ADVISERS.get(body.agent)
    text = body.message.strip()
    if agent is None:
        raise HTTPException(400, "Unknown agent.")
    if not S["active"]:
        raise HTTPException(409, "No session is running.")
    if not text:
        raise HTTPException(400, "Empty message.")
    if len(text) > MAX_MESSAGE:
        raise HTTPException(400, f"Message too long (max {MAX_MESSAGE} characters).")
    session_id = S["session_id"]
    chats = S["chats"][body.agent]
    chats.append({"who": "you", "text": text, "wall": now(), "agent": body.agent})
    eventlog.log("chat_sent", agent=body.agent, sender="participant", recipient=body.agent, text=text)
    started = time.monotonic()
    try:
        reply = await agent.ask(text)
    except Exception as exc:
        eventlog.log("error", agent=body.agent, text=f"chat failed: {type(exc).__name__}: {exc}")
        chats.append({"who": "system", "text": "Agent unavailable. Please try again.", "wall": now(),
                      "agent": body.agent})
        return {"ok": False, "error": "Agent unavailable. Please try again."}
    seconds = round(time.monotonic() - started, 2)
    if not reply:
        eventlog.log("empty_reply", agent=body.agent, seconds=seconds, text="the agent gave no answer")
        reply = "(No answer received. Please ask again.)"
    eventlog.log("chat_reply", agent=body.agent, sender=body.agent, recipient="participant", seconds=seconds,
                 text=reply)
    odd = unverified_numbers(reply, S["blocks"].get(body.agent, [])) if S["session_id"] == session_id else []
    if odd:                                            # logged for the researcher; the participant is not interrupted
        eventlog.log("reply_check", agent=body.agent, text="numbers not in this agent's data: " + ", ".join(odd))
    if S["session_id"] == session_id:
        chats.append({"who": "agent", "text": reply, "wall": now(), "agent": body.agent})
    return {"ok": True, "reply": reply}


# ---------------------------------------------------------------- start-up
def build_agents():
    pin_model()                                      # fixed temperature, no silent fallback model, empty-answer retry
    manager = AgentManager(db_path="submarine_human.db")
    eventlog.install(manager)                        # model calls are logged by the active session's log files
    limit_messages(manager)                          # safety net: drops any agent-to-agent message
    manager.get_agent_messages = lambda *args, **kwargs: ""     # advisers never see other conversations
    for agent in create_advisers():
        manager.register_agent(agent)
        agent.activate()
        ADVISERS[agent.name] = agent


def main():
    build_agents()
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    print(f"Model: {os.getenv('MAHILO_LLM_MODEL', '(Mahilo default)')}   prompts: {prompt_version()}")
    print(f"Open http://{host}:{port}  (opening the browser now)", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()