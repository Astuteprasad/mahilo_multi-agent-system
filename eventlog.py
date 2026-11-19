"""Event log for the experiment.

Every event is written with a time stamp to three files:
  run_log.jsonl    master record, one JSON object per line (full texts)
  run_log.csv      the same events as a table you can open in Excel (one row per event)
  per_second.csv   one row per real second: what each agent did in that second

Columns of run_log.csv
  elapsed         real seconds since the feed started (0 before it starts)
  wall            clock time of the event
  t               simulation step that had been released when the event happened
  agent           the agent the event belongs to (sender for a message)
  event           feed_start, feed_release, queue_depth, agent_start, agent_done, model_call,
                  message_sent, terminal_reply, decision, feed_end
  sender, recipient   for message_sent
  steps           data step(s) the agent was handling, e.g. 5 or 3-6 (agent_start / agent_done)
  seconds         how long the call took
  wait_seconds    how long it waited for the agent's lock before starting
  feed_queue      steps waiting to be given to the agent (queue_depth)
  broker_pending  agent-to-agent messages waiting for the agent (queue_depth)
  tokens          tokens used by a model call
  text            the content: data block, reply, message, decision
  tools           tool calls the model asked for (model_call)
"""
import contextvars
import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

JSONL = Path(os.getenv("SUB_LOG_JSONL", "run_log.jsonl"))
CSV_PATH = Path(os.getenv("SUB_LOG_CSV", "run_log.csv"))
PER_SECOND = Path(os.getenv("SUB_LOG_PER_SECOND", "per_second.csv"))

COLUMNS = ["elapsed", "wall", "t", "agent", "event", "sender", "recipient", "steps", "seconds",
           "wait_seconds", "feed_queue", "broker_pending", "tokens", "text", "tools", "participant"]
AGENTS = ["sonar_operator", "periscope_operator", "track_motion_analyst", "track_manager"]

current_agent = contextvars.ContextVar("current_agent", default="")   # set while an agent's call runs

_lock = threading.Lock()
_state = {"t0": None, "t": None}
PARTICIPANT = ""                              # set by open_session(); written on every row


def open_session(folder, participant) -> None:
    """Human experiment: send all log files of ONE session to its own folder (no wipe of older sessions)."""
    global JSONL, CSV_PATH, PER_SECOND, PARTICIPANT
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    JSONL, CSV_PATH, PER_SECOND = folder / "run_log.jsonl", folder / "run_log.csv", folder / "per_second.csv"
    PARTICIPANT = participant
    reset()                                   # creates the empty files of the new session


def start_clock() -> None:
    """Time zero: called when the first data step is released."""
    _state["t0"] = time.monotonic()


def set_step(t) -> None:
    _state["t"] = t


def elapsed() -> float:
    return 0.0 if _state["t0"] is None else time.monotonic() - _state["t0"]


def reset() -> None:
    """Empty the log files and stop the clock. Called once when the server starts."""
    _state["t0"] = None
    _state["t"] = None
    with _lock:
        JSONL.write_text("", encoding="utf-8")
        with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerow(COLUMNS)


def _one_line(value) -> str:
    return " | ".join(str(value).split("\n")) if value is not None else ""


def log(event: str, agent: str = "", console: bool = True, **fields) -> None:
    row = {"elapsed": round(elapsed(), 3),
           "wall": datetime.now().isoformat(timespec="milliseconds"),
           "t": "" if _state["t"] is None else _state["t"],
           "agent": agent, "event": event, "participant": PARTICIPANT}
    extra = []
    for key, value in fields.items():
        if key in COLUMNS:
            row[key] = value
        else:
            extra.append(f"{key}={value}")
    if extra:
        row["text"] = (str(row.get("text", "")) + " " + " ".join(extra)).strip()
    with _lock:
        with open(JSONL, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as handle:
            csv.writer(handle).writerow(
                [_one_line(row.get(c, ""))[:2000] if c in ("text", "tools") else row.get(c, "") for c in COLUMNS])
    if console:
        text = _one_line(row.get("text", ""))
        text = text if len(text) <= 110 else text[:107] + "..."
        print(f"[{row['elapsed']:7.2f}s] t={row['t']!s:<3} {event:<14} {agent:<21} {text}", flush=True)


def _read_rows():
    rows = []
    if JSONL.exists():
        with open(JSONL, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def write_per_second() -> None:
    """Rebuild per_second.csv from the master log: one row per real second."""
    rows = _read_rows()
    if not rows:
        return
    last = int(max(r["elapsed"] for r in rows))
    buckets = {s: [] for s in range(last + 1)}
    for r in rows:
        buckets[int(r["elapsed"])].append(r)
    header = ["second", "t_released", "events", "decisions"]
    for a in AGENTS:
        header += [f"{a}:calls_done", f"{a}:avg_call_s", f"{a}:messages_sent", f"{a}:messages_received",
                   f"{a}:feed_queue", f"{a}:broker_pending"]
    out = []
    carried_t = ""
    for second in range(last + 1):
        events = buckets[second]
        ts = [r["t"] for r in events if r["t"] != ""]
        if ts:
            carried_t = max(ts)
        line = [second, carried_t, len(events),
                " || ".join(_one_line(r.get("text", "")) for r in events if r["event"] == "decision")]
        for a in AGENTS:
            done = [r for r in events if r["event"] == "agent_done" and r["agent"] == a]
            durations = [float(r["seconds"]) for r in done if r.get("seconds") not in (None, "")]
            queue_rows = [r for r in events if r["event"] == "queue_depth" and r["agent"] == a]
            line += [len(done),
                     round(sum(durations) / len(durations), 2) if durations else "",
                     sum(1 for r in events if r["event"] == "message_sent" and r.get("sender") == a),
                     sum(1 for r in events if r["event"] == "message_sent" and r.get("recipient") == a),
                     queue_rows[-1].get("feed_queue", "") if queue_rows else "",
                     queue_rows[-1].get("broker_pending", "") if queue_rows else ""]
        out.append(line)
    tmp = PER_SECOND.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(out)
    try:
        os.replace(tmp, PER_SECOND)
    except PermissionError:          # the file is open in Excel; try again next time
        pass


def install(manager) -> None:
    """Hook the two places every model call and every agent-to-agent message pass through.
    Mahilo itself is not modified."""
    from mahilo.llm_config import llm_config

    original_call = llm_config.chat_completion

    async def logged_call(*args, **kwargs):
        started = time.monotonic()
        agent = current_agent.get()
        try:
            response = await original_call(*args, **kwargs)
        except Exception as exc:
            log("model_call", agent=agent, seconds=round(time.monotonic() - started, 2),
                text=f"ERROR {type(exc).__name__}: {exc}")
            raise
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        names = []
        for call in calls:
            names.append(f"{call.function.name}({call.function.arguments})")
            if agent == "track_manager" and call.function.name == "contact_human":
                try:
                    text = json.loads(call.function.arguments).get("message", "")
                except Exception:
                    text = call.function.arguments
                if "DECISION" in text.upper():
                    log("decision", agent=agent, text=text)
        usage = getattr(response, "usage", None)
        log("model_call", agent=agent, console=False, seconds=round(time.monotonic() - started, 2),
            tokens=getattr(usage, "total_tokens", "") if usage else "",
            text=message.content or "", tools=" ; ".join(names))
        return response

    llm_config.chat_completion = logged_call

    original_send = manager.send_message_to_agent

    async def logged_send(sender, recipient, message, **kwargs):
        log("message_sent", agent=sender, sender=sender, recipient=recipient, text=message)
        return await original_send(sender=sender, recipient=recipient, message=message, **kwargs)

    manager.send_message_to_agent = logged_send