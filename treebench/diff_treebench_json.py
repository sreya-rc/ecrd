import json
import sys

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def short(x, n=160):
    x = (x or "").replace("\n", " ")
    return x[:n] + ("..." if len(x) > n else "")

def main(a_path, b_path):
    a = load(a_path)
    b = load(b_path)

    ra = a["results"]
    rb = b["results"]

    if len(ra) != len(rb):
        print(f"Different lengths: {len(ra)} vs {len(rb)}")
        sys.exit(1)

    changed_pred = []
    same_pred_diff_trace = []
    acc_a = 0
    acc_b = 0

    for i, (xa, xb) in enumerate(zip(ra, rb)):
        gold_a = (xa.get("answer") or "").upper()
        gold_b = (xb.get("answer") or "").upper()
        if gold_a != gold_b:
            print(f"Gold mismatch at index {i}: {gold_a} vs {gold_b}")
            sys.exit(1)

        pa = (xa.get("prediction") or "").upper()
        pb = (xb.get("prediction") or "").upper()

        if pa == gold_a:
            acc_a += 1
        if pb == gold_b:
            acc_b += 1

        ta = xa.get("trace", {}) or {}
        tb = xb.get("trace", {}) or {}

        trace_fields = [
            "has_answer_tag",
            "num_steps",
            "num_triggered_steps",
            "decider_calls",
            "decider_successes",
        ]
        trace_diff = {
            k: (ta.get(k), tb.get(k))
            for k in trace_fields
            if ta.get(k) != tb.get(k)
        }

        if pa != pb:
            changed_pred.append({
                "idx": i,
                "category": xa.get("category"),
                "gold": gold_a,
                "pred_a": pa,
                "pred_b": pb,
                "correct_a": pa == gold_a,
                "correct_b": pb == gold_b,
                "trace_diff": trace_diff,
                "raw_a": short(xa.get("raw_output", "")),
                "raw_b": short(xb.get("raw_output", "")),
            })
        elif trace_diff:
            same_pred_diff_trace.append({
                "idx": i,
                "category": xa.get("category"),
                "gold": gold_a,
                "pred": pa,
                "trace_diff": trace_diff,
            })

    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print(f"ACC A: {acc_a}/{len(ra)} = {100*acc_a/len(ra):.2f}")
    print(f"ACC B: {acc_b}/{len(rb)} = {100*acc_b/len(rb):.2f}")
    print(f"PREDICTION CHANGES: {len(changed_pred)}")
    print(f"SAME PRED, DIFF TRACE: {len(same_pred_diff_trace)}")

    if changed_pred:
        print("\nPrediction-changing examples:")
        for ex in changed_pred[:20]:
            print(
                f"\n[idx={ex['idx']}] cat={ex['category']} gold={ex['gold']} "
                f"A={ex['pred_a']} ({ex['correct_a']}) B={ex['pred_b']} ({ex['correct_b']})"
            )
            print(f"trace_diff={ex['trace_diff']}")
            print(f"A raw: {ex['raw_a']}")
            print(f"B raw: {ex['raw_b']}")

    if same_pred_diff_trace:
        print("\nSame prediction but different traces:")
        for ex in same_pred_diff_trace[:20]:
            print(f"[idx={ex['idx']}] cat={ex['category']} gold={ex['gold']} pred={ex['pred']}")
            print(f"trace_diff={ex['trace_diff']}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diff_treebench_json.py a.json b.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
