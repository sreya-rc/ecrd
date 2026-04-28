import ast
import argparse
import base64
import csv
import io
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vlm_wrapper_clean import VLMWrapper

csv.field_size_limit(sys.maxsize)


def load_treebench_from_tsv(path: str, max_examples: int = -1) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item = {
                "image": row.get("image", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": row.get("category", ""),
                "multi-choice options": row.get("multi-choice options", ""),
                "target_instances": row.get("target_instances", "[]"),
            }
            data.append(item)
            if max_examples > 0 and len(data) >= max_examples:
                break
    return data


def load_treebench_from_json(path: str, max_examples: int = -1) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    data = []
    for row in rows:
        item = {
            "image": row.get("image", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "category": row.get("category", ""),
            "multi-choice options": row.get("multi-choice options", ""),
            "target_instances": row.get("target_instances", "[]"),
        }
        data.append(item)
        if max_examples > 0 and len(data) >= max_examples:
            break
    return data


def load_treebench(path: str, max_examples: int = -1) -> List[Dict[str, Any]]:
    if path.lower().endswith(".json"):
        return load_treebench_from_json(path, max_examples=max_examples)
    return load_treebench_from_tsv(path, max_examples=max_examples)


def decode_base64_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def compute_box_iou(predict_str: str, target_boxes: list) -> float:
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    all_boxes = []
    for match in matches:
        box = match.strip()
        coord_pattern = r"\[(\d+),(\d+),(\d+),(\d+)\]"
        coord_match = re.match(coord_pattern, box)
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            if x1 < x2 and y1 < y2:
                all_boxes.append([x1, y1, x2, y2])

    def compute_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    if len(target_boxes) == 0 or len(all_boxes) == 0:
        return 0.0

    total_iou = 0.0
    for t_box in target_boxes:
        best_iou = 0.0
        for p_box in all_boxes:
            best_iou = max(best_iou, compute_iou(t_box, p_box))
        total_iou += best_iou

    return total_iou / len(target_boxes)


def summarize_results(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    tags = [
        "Perception/Attributes",
        "Perception/Material",
        "Perception/Physical State",
        "Perception/Object Retrieval",
        "Perception/OCR",
        "Reasoning/Perspective Transform",
        "Reasoning/Ordering",
        "Reasoning/Contact and Occlusion",
        "Reasoning/Spatial Containment",
        "Reasoning/Comparison",
    ]
    alias_map = {"OCR": "Perception/OCR"}

    results = {tag: {"correct": 0, "total": 0} for tag in tags}
    total = 0
    correct = 0

    for item in data:
        category = alias_map.get(item.get("category", ""), item.get("category", ""))
        if category not in results:
            continue
        results[category]["total"] += 1
        total += 1
        if item["prediction"].upper() == item["answer"].upper():
            results[category]["correct"] += 1
            correct += 1

    summary = {
        "overall_correct": correct,
        "overall_total": total,
        "overall_acc": (correct / total * 100.0) if total > 0 else 0.0,
        "mean_iou": float(np.mean([x["iou"] for x in data])) * 100.0 if len(data) > 0 else 0.0,
        "per_category": {},
    }

    for tag in tags:
        cat_total = results[tag]["total"]
        cat_correct = results[tag]["correct"]
        acc = (cat_correct / cat_total * 100.0) if cat_total > 0 else 0.0
        summary["per_category"][tag] = {
            "correct": cat_correct,
            "total": cat_total,
            "acc": acc,
        }

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    for tag, stats in summary["per_category"].items():
        print(f"{tag}: {stats['correct']}/{stats['total']} = {stats['acc']:.2f}")
    print(f"==> Overall: {summary['overall_correct']}/{summary['overall_total']} = {summary['overall_acc']:.2f}")
    print(f"==> Mean IoU: {summary['mean_iou']:.2f}")


def parse_options(options_text: str) -> Dict[str, str]:
    options_text = options_text.strip()
    pattern = r"(?ms)^\s*([A-Z])[\.\:\)]\s*(.*?)(?=^\s*[A-Z][\.\:\)]\s*|\Z)"
    matches = re.findall(pattern, options_text)
    options = {}
    for letter, content in matches:
        options[letter.upper()] = content.strip()

    if len(options) < 2:
        lines = [ln.strip() for ln in options_text.splitlines() if ln.strip()]
        guessed = {}
        for ln in lines:
            m = re.match(r"^\s*([A-Z])[\.\:\)]\s*(.*)$", ln)
            if m:
                guessed[m.group(1).upper()] = m.group(2).strip()
        options = guessed

    return options


def get_valid_letters(option_map: Dict[str, str]) -> List[str]:
    return sorted([k for k in option_map.keys() if len(k) == 1 and k.isalpha()])


def format_options(option_map: Dict[str, str], letters: Optional[List[str]] = None) -> str:
    if letters is None:
        letters = sorted(option_map.keys())
    return "\n".join([f"{k}. {option_map[k]}" for k in letters])


def build_main_prompt(item: Dict[str, Any], option_map: Dict[str, str]) -> str:
    question = item["question"]
    valid_letters = get_valid_letters(option_map)
    options_text = format_options(option_map, valid_letters)
    allowed = ", ".join(valid_letters)
    return (
        "Answer the multiple-choice question using the image.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Think step by step using only visible evidence from the image.\n"
        f"End with the exact format <answer>X</answer> where X is one of {allowed}."
    )


def build_global_description_prompt(item: Dict[str, Any], option_map: Dict[str, str]) -> str:
    return (
        "Describe the image briefly and factually in 1-2 sentences. "
        "Mention only directly visible objects, attributes, text, and spatial relations. "
        "Do not guess. Do not answer the question."
    )


def normalize_answer_letter(text: str, valid_letters: List[str]) -> str:
    text = (text or "").strip().upper()
    valid_set = {x for x in valid_letters if len(x) == 1 and x.isalpha()}

    m = re.search(r"<ANSWER>\s*([A-Z])\s*</ANSWER>", text)
    if m and m.group(1) in valid_set:
        return m.group(1)

    m = re.search(r"ANSWER\s*[:：]\s*([A-Z])", text)
    if m and m.group(1) in valid_set:
        return m.group(1)

    for ch in re.findall(r"\b([A-Z])\b", text):
        if ch in valid_set:
            return ch

    tail = text[-40:]
    for ch in re.findall(r"[A-Z]", tail):
        if ch in valid_set:
            return ch

    return ""


def has_answer_tag(text: str, valid_letters: List[str]) -> bool:
    if not text:
        return False
    valid_set = [x for x in valid_letters if len(x) == 1 and x.isalpha()]
    if not valid_set:
        return False
    allowed = "".join(re.escape(x) for x in valid_set)
    return bool(re.search(rf"<answer>\s*([{allowed}])\s*</answer>", text, flags=re.IGNORECASE))


@dataclass
class EvidenceItem:
    text: str
    source: str = "global"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodingStepInfo:
    step_idx: int
    prefix_text: str
    candidate_tokens: List[str]
    candidate_token_ids: List[int]
    base_probs: List[float]
    evidence_scores: List[float]
    evidence_dist: List[float]
    mixed_probs: List[float]
    chosen_token: str
    chosen_token_id: int
    evidence_pool_size: int
    negotiated_gap: float
    triggered_decider: bool
    decider_evidence: str = ""


@dataclass
class ECRDConfig:
    device: str = "cuda"
    max_new_tokens: int = 64
    max_scan_k: int = 24
    max_evidence_prefixes_per_item: int = 6
    min_evidence_prob: float = 1e-12
    delta: float = 0.05

    max_decider_calls: int = 12
    decider_prefix_tail_chars: int = 320
    decider_max_new_tokens: int = 64
    decider_model: Optional[str] = None

    max_evidence_items: int = 8
    suppress_direct_answer_letters_first_n_steps: int = 2

    stop_on_eos: bool = True
    ban_eos_at_step0: bool = True
    ban_newline_for_first_n_steps: int = 1

    force_min_k_first_n_steps: int = 3
    force_min_k: int = 3
    force_decider_when_winner_changes_first_n_steps: int = 4

    min_decider_gap: float = 0.01
    max_base_confidence_for_override: float = 0.55
    min_evidence_advantage: float = 1.05
    allow_override_if_rank_leq: int = 2
    require_winner_change: bool = True

    max_override_step: int = 2
    require_decider_same_as_mixed: bool = False

    verbose: bool = False


def safe_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    s = x.sum().clamp_min(eps)
    return x / s


def truncate_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def sample_prefix_positions(length: int, max_prefixes: int) -> List[int]:
    if length <= 0:
        return []
    positions = list(range(1, length + 1))
    if len(positions) <= max_prefixes:
        return positions
    idxs = torch.linspace(0, len(positions) - 1, steps=max_prefixes).long().tolist()
    return [positions[i] for i in idxs]


def select_candidate_set_knee(
    probs: torch.Tensor,
    max_scan_k: int,
    min_candidate_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kscan = min(max_scan_k + 1, probs.numel())
    sorted_probs, sorted_ids = torch.topk(probs, k=kscan, dim=-1)

    if kscan == 1:
        return sorted_ids[:1], sorted_probs[:1]

    gaps = sorted_probs[:-1] - sorted_probs[1:]
    k_star = int(torch.argmax(gaps).item()) + 1
    k_star = max(min_candidate_k, min(k_star, kscan - 1))
    return sorted_ids[:k_star], sorted_probs[:k_star]


def compute_qe_for_token(
    vlm: VLMWrapper,
    image: Image.Image,
    user_prompt: str,
    current_prefix_text: str,
    token_id: int,
    evidence_text: str,
    max_prefixes: int,
    min_prob: float,
) -> float:
    ev_ids = vlm.tokenizer.encode(evidence_text, add_special_tokens=False)
    if len(ev_ids) == 0:
        return min_prob

    prefix_positions = sample_prefix_positions(len(ev_ids), max_prefixes)
    vals = []

    for j in prefix_positions:
        ev_prefix = vlm.decode_text(ev_ids[:j]).strip()

        if current_prefix_text.strip() and ev_prefix:
            assistant_prefix = current_prefix_text.rstrip() + " " + ev_prefix
        elif current_prefix_text.strip():
            assistant_prefix = current_prefix_text
        else:
            assistant_prefix = ev_prefix

        logits = vlm.forward_logits_with_assistant_prefix(
            image=image,
            user_prompt=user_prompt,
            assistant_prefix=assistant_prefix,
        )
        probs = torch.softmax(logits, dim=-1)
        p = float(probs[token_id].item())
        vals.append(max(p, min_prob))

    return max(sum(vals) / len(vals), min_prob)


def compute_evidence_distribution(
    vlm: VLMWrapper,
    image: Image.Image,
    user_prompt: str,
    current_prefix_text: str,
    candidate_token_ids: List[int],
    evidence_pool: List[EvidenceItem],
    max_prefixes_per_item: int,
    min_prob: float,
    device: torch.device,
) -> Tuple[List[float], torch.Tensor]:
    if len(evidence_pool) == 0:
        raw = [1.0 for _ in candidate_token_ids]
        ev = torch.ones(len(candidate_token_ids), dtype=torch.float32, device=device)
        ev = safe_normalize(ev)
        return raw, ev

    raw_supports = []
    for tid in candidate_token_ids:
        q_vals = [
            compute_qe_for_token(
                vlm=vlm,
                image=image,
                user_prompt=user_prompt,
                current_prefix_text=current_prefix_text,
                token_id=tid,
                evidence_text=e.text,
                max_prefixes=max_prefixes_per_item,
                min_prob=min_prob,
            )
            for e in evidence_pool
        ]
        mean_q = max(sum(q_vals) / len(q_vals), min_prob)
        raw_supports.append(mean_q)

    ev = torch.tensor(raw_supports, dtype=torch.float32, device=device)
    ev = safe_normalize(ev)
    return raw_supports, ev


def negotiated_reweight_ecrd(candidate_probs: torch.Tensor, evidence_dist: torch.Tensor) -> Tuple[torch.Tensor, float]:
    pi_ci = safe_normalize(candidate_probs.float().clone())
    ri_ci = safe_normalize(evidence_dist.float().clone().to(pi_ci.device))

    alpha = float(pi_ci.max().item())
    pmix_ci = alpha * pi_ci + (1.0 - alpha) * ri_ci
    pmix_ci = safe_normalize(pmix_ci)

    return pmix_ci, alpha


def compute_negotiated_gap(mixed_probs: torch.Tensor) -> float:
    if mixed_probs.numel() <= 1:
        return 1.0
    top2 = torch.topk(mixed_probs, k=2).values
    return float(top2[0].item() - top2[1].item())

def rank_of_index(probs: torch.Tensor, idx: int) -> int:
    order = torch.argsort(probs, descending=True)
    for rank, j in enumerate(order.tolist(), 1):
        if j == idx:
            return rank
    return len(order) + 1


def top2_gap(probs: torch.Tensor) -> float:
    if probs.numel() <= 1:
        return 1.0
    vals = torch.topk(probs, k=2).values
    return float(vals[0].item() - vals[1].item())


def should_trigger_decider(
    *,
    step_idx: int,
    candidate_count: int,
    decider_calls: int,
    base_probs_t: torch.Tensor,
    mixed_probs_t: torch.Tensor,
    config: ECRDConfig,
) -> Tuple[bool, bool, float]:
    if candidate_count <= 1:
        return False, False, 1.0

    if decider_calls >= config.max_decider_calls:
        return False, False, 1.0

    negotiated_gap = compute_negotiated_gap(mixed_probs_t)
    base_top_idx = int(torch.argmax(base_probs_t).item())
    mixed_top_idx = int(torch.argmax(mixed_probs_t).item())
    winner_changed = (base_top_idx != mixed_top_idx)

    gap_condition = (config.min_decider_gap <= negotiated_gap <= config.delta)

    forced_change_condition = (
        step_idx < config.force_decider_when_winner_changes_first_n_steps
        and winner_changed
    )

    if config.require_winner_change:
        trigger = forced_change_condition or (gap_condition and winner_changed)
    else:
        trigger = forced_change_condition or gap_condition

    return trigger, winner_changed, negotiated_gap


def should_accept_decider_override(
    *,
    step_idx: int,
    decider_local_idx: int,
    base_probs_t: torch.Tensor,
    evidence_dist_t: torch.Tensor,
    mixed_probs_t: torch.Tensor,
    config: ECRDConfig,
) -> Tuple[bool, str]:
    base_top_idx = int(torch.argmax(base_probs_t).item())
    mixed_top_idx = int(torch.argmax(mixed_probs_t).item())
    base_conf = float(base_probs_t[base_top_idx].item())

    # NEW: block weak decider confidence
    decider_conf = float(mixed_probs_t[decider_local_idx].item())
    if decider_conf < 0.25:
        return False, "low_decider_conf"

    if step_idx > config.max_override_step and decider_local_idx != mixed_top_idx:
        return False, "too_late"

    if config.require_decider_same_as_mixed and decider_local_idx != mixed_top_idx:
        return False, "not_same_as_mixed"

    if decider_local_idx == mixed_top_idx:
        return True, "same_as_mixed"

    if base_conf >= config.max_base_confidence_for_override and decider_local_idx != base_top_idx:
        return False, "base_too_confident"

    mixed_rank = rank_of_index(mixed_probs_t, decider_local_idx)
    base_rank = rank_of_index(base_probs_t, decider_local_idx)
    if mixed_rank > config.allow_override_if_rank_leq and base_rank > config.allow_override_if_rank_leq:
        return False, "rank_too_low"

    decider_ev = float(evidence_dist_t[decider_local_idx].item())
    base_ev = float(evidence_dist_t[base_top_idx].item())

    score = decider_conf * decider_ev
    base_score = base_conf * base_ev

    margin = score / (base_score + 1e-6)

    # HARD precision gating (paper-like)
    if (
        margin > 1.4              # stronger margin
        and decider_conf > 0.5    # higher confidence
        and decider_ev > base_ev
        and base_conf < 0.55      # don't override confident baseline
    ):
        return True, "paper_like_override"

    return False, "reject"


class ECRDDecoder:
    def __init__(self, base_vlm: VLMWrapper, config: ECRDConfig):
        self.base_vlm = base_vlm
        self.config = config
        self.decider_calls = 0
        self.decider_successes = 0

        if config.decider_model:
            self.decider_vlm = VLMWrapper(config.decider_model, device=config.device)
        else:
            self.decider_vlm = base_vlm

    def init_evidence_pool(self, image: Image.Image, item: Dict[str, Any], option_map: Dict[str, str]) -> List[EvidenceItem]:
        global_desc = self.base_vlm.generate_text(
            image=image,
            prompt=build_global_description_prompt(item, option_map),
            max_new_tokens=32,
            do_sample=False,
        ).strip()
        return [EvidenceItem(text=global_desc, source="global")]

    @torch.no_grad()
    def step_base_distribution(
        self,
        image: Image.Image,
        user_prompt: str,
        prefix_text: str,
        step_idx: int,
        valid_letters: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.base_vlm.forward_logits_with_assistant_prefix(
            image=image,
            user_prompt=user_prompt,
            assistant_prefix=prefix_text,
        )
        probs = F.softmax(logits, dim=-1).clone()

        eos_token_id = getattr(self.base_vlm.tokenizer, "eos_token_id", None)
        if self.config.ban_eos_at_step0 and step_idx == 0 and eos_token_id is not None:
            probs[eos_token_id] = 0.0

        if self.config.ban_newline_for_first_n_steps > 0 and step_idx < self.config.ban_newline_for_first_n_steps:
            newline_ids = []
            for candidate in ["\n", "\n\n"]:
                ids = self.base_vlm.tokenizer.encode(candidate, add_special_tokens=False)
                if len(ids) == 1:
                    newline_ids.append(ids[0])
            for nid in newline_ids:
                probs[nid] = 0.0

        if step_idx < self.config.suppress_direct_answer_letters_first_n_steps:
            for letter in valid_letters:
                ids = self.base_vlm.tokenizer.encode(letter, add_special_tokens=False)
                if len(ids) == 1:
                    probs[ids[0]] *= 0.05

        probs = safe_normalize(probs)

        min_k = 1
        if step_idx < self.config.force_min_k_first_n_steps:
            min_k = self.config.force_min_k

        candidate_ids, candidate_probs = select_candidate_set_knee(
            probs=probs,
            max_scan_k=self.config.max_scan_k,
            min_candidate_k=min_k,
        )
        return probs, candidate_ids, candidate_probs

    def build_decider_prompt(self, item, prefix_text, candidate_tokens):
        tail = truncate_text(prefix_text, self.config.decider_prefix_tail_chars)
        option_lines = [f"{i}: {repr(tok)}" for i, tok in enumerate(candidate_tokens)]
        option_block = "\n".join(option_lines)

        return (
            "You are a careful visual reasoning assistant.\n\n"

            "Step 1: Analyze the image and describe relevant visual evidence.\n"
            "Step 2: Compare all candidate tokens.\n"
            "Step 3: Choose the best next token.\n\n"

            f"Partial answer so far:\n{tail}\n\n"

            f"Candidate tokens:\n{option_block}\n\n"

            "Output EXACTLY:\n"
            "REASONING: <your reasoning>\n"
            "INDEX: <number>\n"
            "EVIDENCE: <one factual visual sentence>"
        )

    def parse_decider_output(self, text: str, num_options: int) -> Tuple[Optional[int], str]:
        idx = None
        ev = ""

        m = re.search(r"INDEX:\s*([0-9]+)", text, flags=re.IGNORECASE)
        if m:
            idx_val = int(m.group(1))
            if 0 <= idx_val < num_options:
                idx = idx_val

        m2 = re.search(r"EVIDENCE:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            ev = m2.group(1).strip()

        if not ev:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                ev = lines[-1]

        return idx, ev

    def call_visual_decider(
        self,
        image: Image.Image,
        item: Dict[str, Any],
        prefix_text: str,
        candidate_token_ids: List[int],
        candidate_tokens: List[str],
    ) -> Tuple[Optional[int], Optional[str], str]:
        prompt = self.build_decider_prompt(item, prefix_text, candidate_tokens)

        out = self.decider_vlm.generate_text(
            image=image,
            prompt=prompt,
            max_new_tokens=self.config.decider_max_new_tokens,
            do_sample=False,
        )

        idx, evidence = self.parse_decider_output(out, len(candidate_tokens))

        if idx is None:
            return None, None, ""

        if evidence is None or len(evidence.strip()) < 5:
            evidence = ""

        return candidate_token_ids[idx], candidate_tokens[idx], evidence

    def decode(self, image: Image.Image, item: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        option_map = parse_options(item.get("multi-choice options", ""))
        valid_letters = get_valid_letters(option_map)
        user_prompt = build_main_prompt(item, option_map)

        evidence_pool = self.init_evidence_pool(image=image, item=item, option_map=option_map)
        generated_ids = []
        step_infos = []

        eos_token_id = getattr(self.base_vlm.tokenizer, "eos_token_id", None)

        for step_idx in range(self.config.max_new_tokens):
            prefix_text = self.base_vlm.decode_text(generated_ids)

            _, candidate_ids_t, candidate_probs_t = self.step_base_distribution(
                image=image,
                user_prompt=user_prompt,
                prefix_text=prefix_text,
                step_idx=step_idx,
                valid_letters=valid_letters,
            )

            candidate_token_ids = candidate_ids_t.tolist()
            candidate_tokens = [self.base_vlm.decode_token(tid) for tid in candidate_token_ids]

            raw_evidence_scores, evidence_dist_t = compute_evidence_distribution(
                vlm=self.base_vlm,
                image=image,
                user_prompt=user_prompt,
                current_prefix_text=prefix_text,
                candidate_token_ids=candidate_token_ids,
                evidence_pool=evidence_pool,
                max_prefixes_per_item=self.config.max_evidence_prefixes_per_item,
                min_prob=self.config.min_evidence_prob,
                device=candidate_probs_t.device,
            )

            mixed_probs_t, _ = negotiated_reweight_ecrd(
                candidate_probs=candidate_probs_t,
                evidence_dist=evidence_dist_t,
            )

            trigger_decider, winner_changed, negotiated_gap = should_trigger_decider(
                step_idx=step_idx,
                candidate_count=len(candidate_token_ids),
                decider_calls=self.decider_calls,
                base_probs_t=candidate_probs_t,
                mixed_probs_t=mixed_probs_t,
                config=self.config,
            )

            decider_evidence = ""

            if trigger_decider:
                if self.config.verbose:
                    print(
                        f"[DECIDER] step={step_idx} gap={negotiated_gap:.4f} "
                        f"k={len(candidate_token_ids)} winner_changed={winner_changed}"
                    )

                decider_token_id, decider_token, decider_evidence = self.call_visual_decider(
                    image=image,
                    item=item,
                    prefix_text=prefix_text,
                    candidate_token_ids=candidate_token_ids,
                    candidate_tokens=candidate_tokens,
                )
                self.decider_calls += 1

                chosen_idx = int(torch.argmax(mixed_probs_t).item())
                chosen_token_id = candidate_token_ids[chosen_idx]
                chosen_token = candidate_tokens[chosen_idx]

                if decider_token_id is not None and decider_token is not None and decider_token_id in candidate_token_ids:
                    decider_local_idx = candidate_token_ids.index(decider_token_id)

                    accept_override, accept_reason = should_accept_decider_override(
                        step_idx=step_idx,
                        decider_local_idx=decider_local_idx,
                        base_probs_t=candidate_probs_t,
                        evidence_dist_t=evidence_dist_t,
                        mixed_probs_t=mixed_probs_t,
                        config=self.config,
                    )

                    if self.config.verbose:
                        print(
                            f"[DECIDER-ACCEPT] step={step_idx} accept={accept_override} "
                            f"reason={accept_reason} token={repr(decider_token)}"
                        )

                    if accept_override:
                        self.decider_successes += 1
                        chosen_token_id = decider_token_id
                        chosen_token = decider_token

                if decider_evidence and len(decider_evidence.strip()) > 10:
                    evidence_pool.append(
                        EvidenceItem(
                            text=decider_evidence.strip(),
                            source="decider",
                            meta={"step_idx": step_idx},
                        )
                    )
                    evidence_pool = evidence_pool[-self.config.max_evidence_items:]
                else:
                    decider_evidence = ""
            else:
                chosen_idx = int(torch.argmax(mixed_probs_t).item())
                chosen_token_id = candidate_token_ids[chosen_idx]
                chosen_token = candidate_tokens[chosen_idx]

            generated_ids.append(chosen_token_id)
            current_text = self.base_vlm.decode_text(generated_ids)

            step_infos.append(
                DecodingStepInfo(
                    step_idx=step_idx,
                    prefix_text=current_text,
                    candidate_tokens=candidate_tokens,
                    candidate_token_ids=candidate_token_ids,
                    base_probs=[float(x) for x in candidate_probs_t.tolist()],
                    evidence_scores=[float(x) for x in raw_evidence_scores],
                    evidence_dist=[float(x) for x in evidence_dist_t.tolist()],
                    mixed_probs=[float(x) for x in mixed_probs_t.tolist()],
                    chosen_token=chosen_token,
                    chosen_token_id=chosen_token_id,
                    evidence_pool_size=len(evidence_pool),
                    negotiated_gap=negotiated_gap,
                    triggered_decider=trigger_decider,
                    decider_evidence=decider_evidence,
                )
            )

            if has_answer_tag(current_text, valid_letters):
                break

            if self.config.stop_on_eos and eos_token_id is not None and chosen_token_id == eos_token_id:
                break

        final_text = self.base_vlm.decode_text(generated_ids).strip()
        pred = normalize_answer_letter(final_text, valid_letters)

        trace = {
            "decider_calls": self.decider_calls,
            "decider_successes": self.decider_successes,
            "num_steps": len(step_infos),
            "num_triggered_steps": sum(int(s.triggered_decider) for s in step_infos),
            "has_answer_tag": has_answer_tag(final_text, valid_letters),
            "evidence_pool": [
                {"text": e.text, "source": e.source, "meta": e.meta}
                for e in evidence_pool
            ],
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "prefix_text": s.prefix_text,
                    "candidate_tokens": s.candidate_tokens,
                    "candidate_token_ids": s.candidate_token_ids,
                    "base_probs": s.base_probs,
                    "evidence_scores": s.evidence_scores,
                    "evidence_dist": s.evidence_dist,
                    "mixed_probs": s.mixed_probs,
                    "chosen_token": s.chosen_token,
                    "chosen_token_id": s.chosen_token_id,
                    "evidence_pool_size": s.evidence_pool_size,
                    "negotiated_gap": s.negotiated_gap,
                    "triggered_decider": s.triggered_decider,
                    "decider_evidence": s.decider_evidence,
                }
                for s in step_infos
            ],
        }

        return pred, final_text, trace


def run_baseline(vlm: VLMWrapper, image: Image.Image, item: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    option_map = parse_options(item.get("multi-choice options", ""))
    valid_letters = get_valid_letters(option_map)
    prompt = build_main_prompt(item, option_map)

    raw_text = vlm.generate_text(
        image=image,
        prompt=prompt,
        max_new_tokens=64,
        do_sample=False,
    ).strip()

    pred = normalize_answer_letter(raw_text, valid_letters)
    trace = {
        "has_answer_tag": has_answer_tag(raw_text, valid_letters),
        "num_steps": None,
    }
    return pred, raw_text, trace


def evaluate_single_item(item: Dict[str, Any], mode: str, vlm: VLMWrapper, predictor: Optional[ECRDDecoder]) -> Dict[str, Any]:
    image = decode_base64_image(item["image"])

    if mode == "baseline":
        pred, raw_text, trace = run_baseline(vlm, image, item)
    else:
        predictor.decider_calls = 0
        predictor.decider_successes = 0
        pred, raw_text, trace = predictor.decode(image, item)

    try:
        target_boxes = ast.literal_eval(item.get("target_instances", "[]"))
    except Exception:
        target_boxes = []

    iou = compute_box_iou(raw_text, target_boxes)

    out = dict(item)
    out["prediction"] = pred
    out["iou"] = iou
    out["raw_output"] = raw_text
    out["trace"] = trace
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="TreeBench closer-to-paper ECRD with dynamic answer letters")
    parser.add_argument("--mode", type=str, choices=["baseline", "token_ecrd"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--decider_model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--treebench_path", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--max_scan_k", type=int, default=24)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_decider_calls", type=int, default=12)
    parser.add_argument("--decider_max_new_tokens", type=int, default=64)
    parser.add_argument("--max_evidence_prefixes_per_item", type=int, default=6)
    parser.add_argument("--max_evidence_items", type=int, default=8)
    parser.add_argument("--suppress_direct_answer_letters_first_n_steps", type=int, default=2)
    parser.add_argument("--force_min_k_first_n_steps", type=int, default=3)
    parser.add_argument("--force_min_k", type=int, default=3)
    parser.add_argument("--force_decider_when_winner_changes_first_n_steps", type=int, default=4)

    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--min_decider_gap", type=float, default=0.01)
    parser.add_argument("--max_base_confidence_for_override", type=float, default=0.55)
    parser.add_argument("--min_evidence_advantage", type=float, default=1.05)
    parser.add_argument("--allow_override_if_rank_leq", type=int, default=2)
    parser.add_argument("--no_require_winner_change", action="store_true")

    parser.add_argument("--max_override_step", type=int, default=2)
    parser.add_argument("--require_decider_same_as_mixed", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.model}")
    vlm = VLMWrapper(args.model, device=args.device)

    predictor = None
    if args.mode == "token_ecrd":
        predictor = ECRDDecoder(
            base_vlm=vlm,
            config=ECRDConfig(
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                max_scan_k=args.max_scan_k,
                delta=args.delta,
                max_decider_calls=args.max_decider_calls,
                decider_max_new_tokens=args.decider_max_new_tokens,
                decider_model=args.decider_model,
                max_evidence_prefixes_per_item=args.max_evidence_prefixes_per_item,
                max_evidence_items=args.max_evidence_items,
                suppress_direct_answer_letters_first_n_steps=args.suppress_direct_answer_letters_first_n_steps,
                force_min_k_first_n_steps=args.force_min_k_first_n_steps,
                force_min_k=args.force_min_k,
                force_decider_when_winner_changes_first_n_steps=args.force_decider_when_winner_changes_first_n_steps,
                verbose=not args.quiet,
                min_decider_gap=args.min_decider_gap,
                max_base_confidence_for_override=args.max_base_confidence_for_override,
                min_evidence_advantage=args.min_evidence_advantage,
                allow_override_if_rank_leq=args.allow_override_if_rank_leq,
                require_winner_change=not args.no_require_winner_change,
                max_override_step=args.max_override_step,
                require_decider_same_as_mixed=args.require_decider_same_as_mixed,
            ),
        )

    print("Loading TreeBench...")
    ds = load_treebench(args.treebench_path, max_examples=args.max_examples)

    results = []
    for item in tqdm(ds, desc=f"Evaluating ({args.mode})"):
        out = evaluate_single_item(
            item=item,
            mode=args.mode,
            vlm=vlm,
            predictor=predictor,
        )
        results.append(out)

    summary = summarize_results(results)
    print_summary(summary)

    payload = {
        "mode": args.mode,
        "model": args.model,
        "decider_model": args.decider_model,
        "treebench_path": args.treebench_path,
        "summary": summary,
        "results": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()
