"""After a run:  python score_v2.py
Compares the maps the Track Manager showed (run_log.jsonl) with the ground truth: positions and ship types.
Needs, in the same folder: run_log.jsonl, agent_feed.csv, ship_data_all.csv
Writes: map_score.csv (every plotted point, its error and the type answer) and map_score.png (paths over the truth)."""
import csv
import json
import math
import re
import sys
from collections import defaultdict

LOG, FEED, TRUTH = "run_log.jsonl", "agent_feed.csv", "ship_data_all.csv"
SHIP_OF = {"Contact_1": "Ship_1", "Contact_2": "Hostile_1", "Contact_3": "Ferry_1",
           "Contact_4": "Cargo_1", "Contact_5": "Fishing_2", "Contact_6": "Fishing_3"}
LINE = re.compile(r"(Ownship|Contact_\d+)\s*:\s*x\s*=\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*y\s*=\s*(-?\d+(?:\.\d+)?)", re.I)
HEAD = re.compile(r"MAP\s+t\s*=\s*(\d+)", re.I)
TYPE = re.compile(r"(Contact_\d+)\s*:[^\n]*?type\s*=\s*([^\n]+)", re.I)
KINDS = {"fishing vessel", "ferry", "cargo ship", "hostile ship"}


def parse_type(raw):
    """'ferry or hostile ship' -> {'ferry', 'hostile ship'}; 'unknown' -> set(); anything else -> None."""
    text = re.sub(r"[\"'.*()\[\]]", "", raw).strip().lower()
    text = re.sub(r"\s*(?:,|/|\band\b)\s*", " or ", text)
    if text in ("unknown", "unsure", "none", ""):
        return set()
    parts = {p.strip() for p in text.split(" or ") if p.strip()}
    return parts if parts and parts <= KINDS else None


def tm_maps(path):
    """({step: {name: (x, y)}}, {step: {contact: raw type text}}) from the contact_human calls of the Track
    Manager (the last map of a step wins)."""
    maps, types = {}, {}
    decoder = json.JSONDecoder()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "model_call" or row.get("agent") != "track_manager":
            continue
        tools = row.get("tools") or ""
        pos = 0
        while True:
            i = tools.find("contact_human(", pos)
            if i < 0:
                break
            start = i + len("contact_human(")
            try:
                args, end = decoder.raw_decode(tools[start:])
            except ValueError:
                break
            pos = start + end
            text = str(args.get("message", ""))
            head = HEAD.search(text)
            if not head:
                continue
            step = int(head.group(1))
            maps[step] = {m.group(1).title() if m.group(1).lower() == "ownship" else m.group(1).capitalize():
                          (float(m.group(2)), float(m.group(3))) for m in LINE.finditer(text)}
            types[step] = {m.group(1).capitalize(): m.group(2).strip() for m in TYPE.finditer(text)}
    return maps, types


def main():
    feed = list(csv.DictReader(open(FEED, encoding="utf-8-sig")))
    minute_of = {int(r["time_step"]): int(r["time_min"]) for r in feed}
    contacts = sorted({r["contact"] for r in feed})
    truth, kind_of = defaultdict(dict), {}
    for r in csv.DictReader(open(TRUTH, encoding="utf-8-sig")):
        t = int(r["time_min"])
        truth[r["ship_name"]][t] = (float(r["ship_x"]), float(r["ship_y"]))
        truth["Ownship"][t] = (float(r["own_x"]), float(r["own_y"]))
        kind_of[r["ship_name"]] = r["ship_type"].strip().lower()
    maps, types = tm_maps(LOG)
    if not maps:
        sys.exit("No map from the Track Manager found in run_log.jsonl (no contact_human call with 'MAP t=...').")

    rows, errs = [], defaultdict(list)
    outcome = defaultdict(dict)                 # contact -> step -> correct / narrowed / wrong / unknown / none
    names = ["Ownship"] + contacts
    for step in sorted(minute_of):
        for name in names:
            ship = "Ownship" if name == "Ownship" else SHIP_OF[name]
            tx, ty = truth[ship][minute_of[step]]
            got = maps.get(step, {}).get(name)
            true_kind = "" if name == "Ownship" else kind_of[ship]
            raw = types.get(step, {}).get(name, "") if name != "Ownship" else ""
            result = ""
            if name != "Ownship":
                if got is None or not raw:
                    result = "none"
                else:
                    kinds = parse_type(raw)
                    if kinds is None:
                        result = "invalid"
                    elif not kinds:
                        result = "unknown"
                    elif kinds == {true_kind}:
                        result = "correct"
                    elif true_kind in kinds:
                        result = "narrowed"
                    else:
                        result = "wrong"
                outcome[name][step] = result
            if got is None:
                rows.append([step, minute_of[step], name, tx, ty, "", "", "", true_kind, raw, result])
            else:
                e = math.hypot(got[0] - tx, got[1] - ty)
                errs[name].append(e)
                rows.append([step, minute_of[step], name, tx, ty, got[0], got[1], round(e, 3), true_kind, raw, result])
    with open("map_score.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["step", "time_min", "name", "true_x", "true_y", "tm_x", "tm_y", "error_cells",
                    "true_type", "tm_type", "type_result"])
        w.writerows(rows)

    steps = len(minute_of)
    last = max(minute_of)
    print(f"Steps with a map from the Track Manager: {len(maps)} of {steps}\n")
    print("POSITIONS")
    print(f"{'name':<10} {'plotted':>8} {'mean err':>9} {'max err':>8} {'within 0.5':>11} {'within 1':>9}")
    allerr = []
    for name in names:
        e = errs[name]
        allerr += e if name != "Ownship" else []
        if e:
            print(f"{name:<10} {len(e):>4}/{steps:<3} {sum(e)/len(e):>9.2f} {max(e):>8.2f} "
                  f"{100*sum(x <= 0.5 for x in e)/len(e):>10.0f}% {100*sum(x <= 1 for x in e)/len(e):>8.0f}%")
        else:
            print(f"{name:<10} {0:>4}/{steps:<3}   (nothing plotted)")
    if allerr:
        print(f"\nAll contacts: mean error {sum(allerr)/len(allerr):.2f} cells, "
              f"{100*sum(x <= 1 for x in allerr)/len(allerr):.0f}% of the points within 1 cell, "
              f"{len(allerr)} of {steps * len(contacts)} points plotted.")

    print("\nSHIP TYPES  (correct = exactly the right type; narrowed = the right type is one of two named;"
          " unknown; wrong)")
    print(f"{'contact':<10} {'true type':<15} {'answer at the last step':<30} {'result':<9} "
          f"{'right from step':>16}  correct/narrowed/unknown/wrong/none")
    final_correct = 0
    for c in contacts:
        true_kind = kind_of[SHIP_OF[c]]
        res = outcome[c]
        counts = {k: sum(1 for v in res.values() if v == k) for k in ("correct", "narrowed", "unknown", "wrong")}
        counts["none"] = sum(1 for v in res.values() if v in ("none", "invalid"))
        stay = None                              # first step from which every later answer is "correct"
        for s in sorted(minute_of):
            if all(res.get(k) == "correct" for k in sorted(minute_of) if k >= s):
                stay = s
                break
        final_raw = types.get(last, {}).get(c, "(no map)")
        final_res = res.get(last, "none")
        final_correct += final_res == "correct"
        print(f"{c:<10} {true_kind:<15} {final_raw:<30} {final_res:<9} {('step ' + str(stay)) if stay is not None else 'never':>16}"
              f"  {counts['correct']}/{counts['narrowed']}/{counts['unknown']}/{counts['wrong']}/{counts['none']}")
    print(f"\nAt the last step: {final_correct} of {len(contacts)} contacts named exactly right.")
    print("Wrote map_score.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9.6))
        ax.set_xlim(0.5, 20.5); ax.set_ylim(20.5, 0.5)
        ax.set_xticks(range(1, 21)); ax.set_yticks(range(1, 21))
        ax.xaxis.tick_top(); ax.grid(True, color="#bbbbbb", linewidth=0.5); ax.set_aspect("equal")
        for i, name in enumerate(names):
            ship = "Ownship" if name == "Ownship" else SHIP_OF[name]
            colour = plt.cm.tab10(i)
            ts = sorted(truth[ship])
            ax.plot([truth[ship][t][0] for t in ts], [truth[ship][t][1] for t in ts], color=colour, lw=1.2, alpha=0.6)
            pts = [maps[s][name] for s in sorted(maps) if name in maps[s]]
            if pts:
                label = f"{name} (track manager)"
                if name != "Ownship":
                    label += f": {types.get(last, {}).get(name, '?')}"
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o--", color=colour, ms=4, lw=1, label=label)
        ax.set_title("Thin line = true route, dots = what the Track Manager plotted")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)
        fig.savefig("map_score.png", dpi=120, bbox_inches="tight")
        print("Wrote map_score.png")
    except ImportError:
        print("(matplotlib not installed: no picture made)")


if __name__ == "__main__":
    main()
