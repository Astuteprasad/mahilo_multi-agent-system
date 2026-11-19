"""Version 2 feeder: reads scenario_feed.csv and gives each agent its own columns, one step at a time.

  sonar_operator      contact, sonar_cue
  track_analyst       own_x, own_y, and contact, track_x, track_y, range, speed   (whole-number positions and ranges)
  periscope_operator  contact, periscope report, collision warning (human experiment only)
  track_manager       nothing (it only receives reports)
Steps are released every TICK_SECONDS (default 20). If an agent is still busy with an earlier step,
the waiting steps are delivered together in one message (same as the earlier feeder)."""
import asyncio
import csv
import os
import time
from collections import defaultdict

import eventlog
from agents import SONAR
from agents_v2 import TA
from feeder import _n, _send, _worker, _sampler       # reused unchanged

TICK = float(os.getenv("TICK_SECONDS", "20"))
START_DELAY = float(os.getenv("FEED_START_DELAY", "10"))
CSV_PATH = os.getenv("FEED_CSV", "scenario_feed.csv")
TAIL_SECONDS = float(os.getenv("FEED_TAIL_SECONDS", "60"))


def load_steps(path):
    steps = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            steps[int(row["time_step"])].append(row)
    return dict(steps)


def sonar_block(step, rows):
    lines = [f'(sonar display) t={step} time_min={rows[0]["time_min"]}']
    lines += [f'{r["contact"]}: cue={r["sonar_cue"]}' for r in rows]
    return "\n".join(lines)


def ta_block(step, rows):
    lines = [f'(motion display) t={step} time_min={rows[0]["time_min"]}',
             f'Ownship: x={_n(rows[0]["own_x"])} y={_n(rows[0]["own_y"])}']
    lines += [f'{r["contact"]}: x={_n(r["track_x"])} y={_n(r["track_y"])} range={_n(r["range"])} speed={_n(r["speed"])}'
              for r in rows]
    return "\n".join(lines)


def periscope_block(step, rows):
    raised = any(r["periscope_raised"].strip().lower() == "yes" for r in rows)
    lines = [f'(periscope display) t={step} time_min={rows[0]["time_min"]} periscope={"raised" if raised else "down"}']
    if not raised:
        lines.append("Periscope is down: no visual report and no collision warning at this step.")
        return "\n".join(lines)
    for r in rows:
        lines.append(f'{r["contact"]}: visual report={r["periscope_report"].strip()} '
                     f'collision warning={r["collision_warning"].strip()}')
    return "\n".join(lines)


BLOCKS = {SONAR: sonar_block, TA: ta_block}


async def run_feed(manager, sockets_for):
    steps = load_steps(CSV_PATH)
    names = [SONAR, TA]
    print("Waiting for the three agents to connect (one terminal per agent)...", flush=True)
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
    for k, step in enumerate(sorted(steps)):
        await asyncio.sleep(max(0.0, start + k * TICK - time.monotonic()))
        eventlog.set_step(step)
        for n in names:
            text = BLOCKS[n](step, steps[step])
            eventlog.log("feed_release", agent=n, console=False, text=text)
            await _send(sockets_for(n), text)          # the operator sees the data at once
            queues[n].put_nowait(text)                  # and the agent receives it
        eventlog.log("feed_release", text=f"step {step} released")
    eventlog.log("feed_end", text="last step released; log keeps running while agents finish")
    await asyncio.sleep(TAIL_SECONDS)
    sampler.cancel()
    eventlog.write_per_second()
    print("Logs written: run_log.csv, per_second.csv, run_log.jsonl. Now run: python score_v2.py", flush=True)