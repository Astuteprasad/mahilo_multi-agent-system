"""After a human session:   python score_human.py logs/<session folder>
Needs, in the current folder: ship_data_2h.csv (the true scenario) and score_v2.py.
Scores, per step and per track:
  - the ship-type answer (correct / unknown / wrong / none),
  - HOSTILE DETECTION: an answer "hostile ship" counts as flagging that track as hostile -> hits, misses,
    false alarms, and the step from which every hostile ship stays flagged,
  - the Distance answer against the true range (rounded to whole cells), and the confidence given,
and shows the threat answers and how the participant used the advisers (questions, delivery checks, data shown).
Writes human_score.csv in the session folder. (Positions come from the paper map and are scored separately.)"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from score_v2 import parse_type

TRUTH = os.getenv("SCENARIO_TRUTH", "ship_data_2h.csv")


def whole(value):
    return int(float(value) + 0.5)


def tn(contact):
    return contact.replace("Contact_", "Track_")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python score_human.py logs\\<session folder>")
    folder = Path(sys.argv[1])
    truth = list(csv.DictReader(open(TRUTH, encoding="utf-8-sig")))
    steps = sorted({int(r["time_step"]) for r in truth})
    contacts = sorted({r["contact"] for r in truth})
    kind_of = {r["contact"]: r["ship_type"].strip().lower() for r in truth}
    name_of = {r["contact"]: r["ship_name"] for r in truth}
    range_at = {(r["contact"], int(r["time_step"])): whole(r["range_cells"]) for r in truth}
    hostile = [c for c in contacts if kind_of[c] == "hostile ship"]

    given = defaultdict(dict)        # step -> contact -> answer row; the last submit of a step wins
    for r in csv.DictReader(open(folder / "answers.csv", encoding="utf-8-sig")):
        given[int(r["step"])][r["contact"]] = r

    rows, outcome, flagged = [], defaultdict(dict), defaultdict(dict)
    for step in steps:
        for c in contacts:
            a = given.get(step, {}).get(c)
            raw = a["type"] if a else ""
            if not raw:
                result = "none"
            else:
                kinds = parse_type(raw)
                if kinds is None:
                    result = "invalid"
                elif not kinds:
                    result = "unknown"
                elif kinds == {kind_of[c]}:
                    result = "correct"
                else:
                    result = "wrong"
            outcome[c][step] = result
            flagged[c][step] = (raw.strip().lower() == "hostile ship") if raw else None
            dist = a["distance"] if a else ""
            err = abs(int(dist) - range_at[(c, step)]) if dist != "" else ""
            rows.append([step, c, name_of[c], kind_of[c], raw, result, dist, range_at[(c, step)], err,
                         a["confidence"] if a else "", a["threat"] if a else ""])
    with open(folder / "human_score.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["step", "contact", "ship_name", "true_type", "answer", "type_result", "distance_answer",
                    "true_range", "distance_error", "confidence", "threat_answer"])
        w.writerows(rows)

    print(f"Session: {folder.name}\n")
    print("SHIP TYPES  (correct / unknown / wrong / none over all steps)")
    print(f"{'track':<10} {'ship':<10} {'true type':<15} {'answer at last step':<20} {'result':<9} {'right from step':>15}  counts")
    final_correct = 0
    last = steps[-1]
    for c in contacts:
        res = outcome[c]
        counts = Counter(res.values())
        stay = next((s for s in steps if all(res[k] == "correct" for k in steps if k >= s)), None)
        final_correct += res[last] == "correct"
        ans = given.get(last, {}).get(c, {}).get("type", "(none)")
        print(f"{tn(c):<10} {name_of[c]:<10} {kind_of[c]:<15} {ans:<20} {res[last]:<9} "
              f"{('step ' + str(stay)) if stay is not None else 'never':>15}  "
              f"{counts['correct']}/{counts['unknown']}/{counts['wrong']}/{counts['none'] + counts['invalid']}")
    print(f"\nAt the last step: {final_correct} of {len(contacts)} tracks named exactly right.")

    names = ", ".join(f"{tn(c)} = {name_of[c]}" for c in hostile)
    print(f"\nHOSTILE DETECTION  (true hostile tracks: {names})")
    print(f"{'step':>4} {'hits':>5} {'misses':>7} {'false alarms':>13}   false alarms are")
    for step in steps:
        if step not in given:
            print(f"{step:>4}   (no answers)")
            continue
        hits = sum(flagged[c][step] is True for c in hostile)
        misses = sum(flagged[c][step] is not True for c in hostile)
        fa = [c for c in contacts if c not in hostile and flagged[c][step]]
        print(f"{step:>4} {hits:>5} {misses:>7} {len(fa):>13}   {', '.join(tn(c) for c in fa) or '-'}")
    for c in hostile:
        first = next((s for s in steps if flagged[c][s] is True), None)
        stable = next((s for s in steps if all(flagged[c][k] is True for k in steps if k >= s)), None)
        print(f"  {tn(c)} ({name_of[c]}): first flagged at step "
              f"{first if first is not None else 'never'}, stably flagged from step {stable if stable is not None else 'never'}")
    both = next((s for s in steps if all(flagged[c][k] is True for c in hostile for k in steps if k >= s)), None)
    print(f"  All hostile tracks stably flagged from step: {both if both is not None else 'never'}")
    total_fa = sum(1 for s in steps if s in given for c in contacts if c not in hostile and flagged[c][s])
    print(f"  False alarms over the whole session: {total_fa} (track-steps)")

    errs = [r[8] for r in rows if r[8] != ""]
    print("\nDISTANCE ANSWERS  (whole cells vs the true range rounded to whole cells)")
    if errs:
        print(f"  {len(errs)} answers, mean absolute error {sum(errs) / len(errs):.2f} cells, "
              f"{sum(e == 0 for e in errs)} exactly right")
    conf = defaultdict(list)
    for r in rows:
        if r[9] != "" and r[5] in ("correct", "wrong"):
            conf[r[5]].append(int(r[9]))
    if conf:
        print("\nCONFIDENCE  (mean 1-5 rating)  " + ",  ".join(
            f"when {k}: {sum(v) / len(v):.2f} ({len(v)} answers)" for k, v in sorted(conf.items())))

    print("\nTHREAT ANSWERS at the last step (not scored until the threat definition is agreed)")
    for c in contacts:
        print(f"  {tn(c)}: {given.get(last, {}).get(c, {}).get('threat', '(none)')}")

    events = [json.loads(l) for l in open(folder / "run_log.jsonl", encoding="utf-8") if l.strip()]
    asked, secs = Counter(), defaultdict(list)
    for e in events:
        if e["event"] == "chat_sent":
            asked[(e["agent"], e["t"])] += 1
        if e["event"] == "chat_reply" and e.get("seconds") not in ("", None):
            secs[e["agent"]].append(float(e["seconds"]))
    print("\nUSE OF THE ADVISERS (questions asked, by step)")
    for agent in sorted({a for a, _ in asked} | set(secs)):
        by_step = {t: n for (a, t), n in sorted(asked.items(), key=lambda kv: str(kv[0][1])) if a == agent}
        total = sum(by_step.values())
        mean = sum(secs[agent]) / len(secs[agent]) if secs[agent] else 0
        print(f"  {agent}: {total} questions, mean reply {mean:.1f}s, by step {by_step}")
    shown = Counter(e["agent"] for e in events if e["event"] == "show_data")
    if shown:
        print("  'Show latest data' used: " + ", ".join(f"{a} x{n}" for a, n in sorted(shown.items())))

    bad = [e for e in events if e["event"] == "delivery_check" and "NOT DELIVERED" in str(e.get("text", ""))]
    odd = [e for e in events if e["event"] == "reply_check"]
    print("\nDATA CHECKS")
    print(f"  Delivery to the agents: {'every step reached every agent' if not bad else str(len(bad)) + ' DELIVERY FAILURES, see run_log.csv'}")
    print(f"  Replies with numbers that are not in the agent's own data: {len(odd)}"
          + ("  (see 'reply_check' rows in run_log.csv)" if odd else ""))

    start = {e["t"]: e["elapsed"] for e in events if e["event"] == "next_pressed"}
    first = {}
    for e in events:
        if e["event"] == "answer_submit":
            first.setdefault(e["t"], e["elapsed"])
    print("\nTIME PER STEP (seconds from release to the first submitted answer)")
    print("  " + ", ".join(f"step {t}: {first[t] - start[t]:.0f}s" for t in sorted(start) if t in first))
    print(f"\nWrote {folder / 'human_score.csv'}")


if __name__ == "__main__":
    main()