"""Reads ship_simulation.csv and gives each agent its own columns, one time step per second.

  sonar_operator       ship_name, true_bearing_deg, rel_bearing_deg
  track_motion_analyst ship_name, speed, x_position, y_position
  periscope_operator   ship_name, Visual details
  track_manager        nothing (it only receives reports)
Each block is shown on that operator's terminal straight away and is also sent to the agent. If an
agent is still busy with an earlier step, the waiting steps are delivered together in one message.
Every release, queue depth and reply is written to the event log (eventlog.py).
"""
import asyncio
import csv
import os
import time
from collections import defaultdict

import eventlog
from agents import PERISCOPE, SONAR, TMA

TICK = float(os.getenv("TICK_SECONDS", "1.0"))
START_DELAY = float(os.getenv("FEED_START_DELAY", "10"))
CSV_PATH = os.getenv("CSV_PATH", "ship_simulation.csv")
TAIL_SECONDS = float(os.getenv("FEED_TAIL_SECONDS", "30"))   # keep logging after the last step


def _n(value):
    return f"{round(float(value), 3):g}"


def load_steps(path):
    steps = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            steps[int(row["time_step"])].append(row)
    return dict(steps)


def sonar_block(t, rows):
    lines = [f"(sonar display) t={t}"]
    lines += [f'{r["ship_name"]}: true_bearing={_n(r["true_bearing_deg"])} rel_bearing={_n(r["rel_bearing_deg"])}' for r in rows]
    return "\n".join(lines)


def tma_block(t, rows):
    lines = [f"(motion display) t={t}"]
    lines += [f'{r["ship_name"]}: speed={_n(r["speed"])} x={_n(r["x_position"])} y={_n(r["y_position"])}' for r in rows]
    return "\n".join(lines)


def periscope_block(t, rows):
    seen = [r for r in rows if (r.get("Visual details") or "").strip()]
    if not seen:
        return None
    lines = [f"(periscope display) t={t}"]
    lines += [f'{r["ship_name"]}: visual={r["Visual details"].strip()}' for r in seen]
    return "\n".join(lines)


BLOCKS = {SONAR: sonar_block, TMA: tma_block, PERISCOPE: periscope_block}


async def _send(sockets, text):
    for ws in sockets:
        try:
            await ws.send_text(text)
        except Exception:
            pass


def _broker_pending(manager, name):
    try:
        return len(manager.message_broker.get_pending_messages(name))
    except Exception:
        return ""


async def _worker(manager, agent, sockets_for, queue):
    """Delivers steps to one agent, one call at a time; steps that piled up go in one message."""
    while True:
        batch = [await queue.get()]
        while not queue.empty():
            batch.append(queue.get_nowait())
        try:
            reply = await agent.process_chat_message("\n".join(batch), websockets=sockets_for(agent.name))
            text = str((reply or {}).get("response", "")).strip()
            if text and text != "ACK":
                eventlog.log("terminal_reply", agent=agent.name, text=text)
                await _send(sockets_for(agent.name), f"{agent.name}: {text}")
        except Exception as exc:
            eventlog.log("error", agent=agent.name, text=f"{type(exc).__name__}: {exc}")


async def _sampler(manager, queues):
    """Once a second: how many steps wait for each agent, and how many messages wait in the broker."""
    while True:
        for a in manager.get_all_agents():
            q = queues.get(a.name)
            eventlog.log("queue_depth", agent=a.name, console=False,
                         feed_queue=q.qsize() if q is not None else "",
                         broker_pending=_broker_pending(manager, a.name))
        eventlog.write_per_second()
        await asyncio.sleep(1.0)


async def run_feed(manager, sockets_for):
    steps = load_steps(CSV_PATH)
    names = [SONAR, TMA, PERISCOPE]
    print("Waiting for the four operators to connect (one terminal per agent)...", flush=True)
    while not all(sockets_for(a.name) for a in manager.get_all_agents()):
        await asyncio.sleep(0.5)
    print(f"All connected. Feed starts in {START_DELAY:g}s: {len(steps)} steps, {TICK:g}s each.", flush=True)
    await asyncio.sleep(START_DELAY)
    queues = {n: asyncio.Queue() for n in names}
    for n in names:
        asyncio.create_task(_worker(manager, manager.get_agent(n), sockets_for, queues[n]))
    eventlog.start_clock()
    eventlog.log("feed_start", text=f"{len(steps)} steps, {TICK:g}s each")
    sampler = asyncio.create_task(_sampler(manager, queues))
    start = time.monotonic()
    for t in sorted(steps):
        await asyncio.sleep(max(0.0, start + t * TICK - time.monotonic()))
        eventlog.set_step(t)
        for n in names:
            text = BLOCKS[n](t, steps[t])
            if text:
                eventlog.log("feed_release", agent=n, console=False, text=text)
                await _send(sockets_for(n), text)       # the operator sees the data at once
                queues[n].put_nowait(text)               # and the agent receives it
        eventlog.log("feed_release", text=f"t={t} released")
    eventlog.log("feed_end", text="last step released; log keeps running while agents finish")
    await asyncio.sleep(TAIL_SECONDS)       # let the agents finish, then close the per-second log
    sampler.cancel()
    eventlog.write_per_second()
    print("Logs written: run_log.csv, per_second.csv, run_log.jsonl", flush=True)