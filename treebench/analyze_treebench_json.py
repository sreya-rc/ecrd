import json
import sys
from statistics import mean

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def short(x, n=180):
    x = (x or "").replace("\n", " ")
    return x[:n] + ("..." if len(x) > n else "")

def analyze(path):
    payload = load(path)
    results = payload["results"]

    correct = 0
    total = len(results)
    has_answer_tag = 0
    step_counts = []
    triggered_counts = []
    decider_calls = []
    decider_successes = []

    failures = []

    for r in results:
        pred = (r.get("prediction") or "").upper()
        gold = (r.get("answer") or "").upper()
        trace = r.get("trace", {}) or {}

        if pred == gold:
            correct += 1

        if trace.get("has_answer_tag") is True:
            has_answer_tag += 1

        if trace.get("num_steps") is not None:
            step_counts.append(trace["num_steps"])

        if trace.get("num_triggered_steps") is not None:
            triggered_counts.append(trace["num_triggered_steps"])

        if trace.get("decider_calls") is not None:
            decider_calls.append(trace["decider_calls"])

        if trace.get("decider_successes") is not None:
            decider_successes.append(trace["decider_successes"])

        if pred != gold:
            failures.append({
                "category": r.get("category"),
                "gold": gold,
                "pred": pred,
                "has_answer_tag": trace.get("has_answer_tag"),
                "num_steps": trace.get("num_steps"),
                "num_triggered_steps": trace.get("num_triggered_steps"),
                "decider_calls": trace.get("decider_calls"),
                "decider_successes": trace.get("decider_successes"),
                "raw_output": r.get("raw_output", ""),
            })

    print(f"\nFILE: {path}")
    print(f"MODE: {payload.get('mode')}")
    print(f"ACC: {correct}/{total} = {100.0 * correct / total:.2f}")
    print(f"HAS_ANSWER_TAG: {has_answer_tag}/{total} = {100.0 * has_answer_tag / total:.2f}")

    if step_counts:
        print(f"AVG_NUM_STEPS: {mean(step_counts):.2f}")
    if triggered_counts:
        print(f"AVG_TRIGGERED_STEPS: {mean(triggered_counts):.2f}")
    if decider_calls:
        print(f"AVG_DECIDER_CALLS: {mean(decider_calls):.2f}")
    if decider_successes:
        print(f"AVG_DECIDER_SUCCESSES: {mean(decider_successes):.2f}")

    print(f"\nFIRST 8 FAILURES:")
    for i, f in enumerate(failures[:8], 1):
        print(f"\n[{i}] category={f['category']} gold={f['gold']} pred={f['pred']}")
        print(
            f"has_answer_tag={f['has_answer_tag']} "
            f"num_steps={f['num_steps']} "
            f"triggered={f['num_triggered_steps']} "
            f"decider_calls={f['decider_calls']} "
            f"decider_successes={f['decider_successes']}"
        )
        print(short(f["raw_output"]))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_treebench_json.py file1.json [file2.json ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        analyze(path)
