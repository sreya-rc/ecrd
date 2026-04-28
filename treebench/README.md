# TreeBench ECRD Tests

This directory contains TreeBench evaluation code for:

- **Baseline** greedy decoding with LLaVA-OneVision-7B
- **ECRD-style decoding** with token-level evidence reweighting and optional GRIT decider

This is a partial reproduction of the ECRD paper.

## Files

```text
treebench_ecrd.py            # baseline + ECRD-style TreeBench runner
vlm_wrapper_clean.py         # VLM wrapper for LLaVA-OneVision / Qwen / GRIT
analyze_treebench_json.py    # summarize output JSON accuracy
diff_treebench_json.py       # compare prediction changes between two runs
treebench_balanced_200.json  # TreeBench balanced subset
```

## Baseline
```text
CUDA_VISIBLE_DEVICES=0 python3 treebench_ecrd.py \
  --mode baseline \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --device cuda \
  --treebench_path ./treebench_balanced_200.json \
  --output_json ./baseline_200.json \
  2>&1 | tee ./baseline_200.log
```

## ECRD
```text
CUDA_VISIBLE_DEVICES=0 python3 treebench_ecrd.py \
  --mode token_ecrd \
  --model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --decider_model /path/to/grit_local/GRIT-20-Qwen2.5-VL-3B \
  --device cuda \
  --treebench_path ./treebench_balanced_200.json \
  --delta 0.05 \
  --min_decider_gap 0.01 \
  --max_scan_k 24 \
  --max_new_tokens 64 \
  --max_decider_calls 12 \
  --decider_max_new_tokens 96 \
  --max_evidence_prefixes_per_item 6 \
  --max_evidence_items 8 \
  --suppress_direct_answer_letters_first_n_steps 2 \
  --force_min_k_first_n_steps 3 \
  --force_min_k 3 \
  --force_decider_when_winner_changes_first_n_steps 4 \
  --max_base_confidence_for_override 0.55 \
  --min_evidence_advantage 1.05 \
  --allow_override_if_rank_leq 2 \
  --output_json ./ecrd_safe_200.json \
  2>&1 | tee ./ecrd_safe_200.log
```
