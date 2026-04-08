import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from vlm_wrapper_pope import VLMWrapper


def load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            raise ValueError("JSON root is a dict but no list found under 'data'.")
        if isinstance(data, list):
            return data
        raise ValueError("Expected JSON list or JSONL records.")
    except json.JSONDecodeError:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def resolve_image_path(row: Dict[str, Any], image_root: Optional[str]) -> str:
    candidates = [
        row.get("image"),
        row.get("image_path"),
        row.get("img"),
        row.get("filename"),
    ]
    rel = next((x for x in candidates if isinstance(x, str) and x.strip()), None)
    if rel is None:
        raise ValueError(f"Could not find image path field in row keys: {list(row.keys())}")

    if os.path.isabs(rel):
        return rel
    if image_root:
        return os.path.join(image_root, rel)
    return rel


def load_pope(path: str, image_root: Optional[str], max_examples: int = -1) -> List[Dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    data = []
    for row in rows:
        item = {
            "image_path": resolve_image_path(row, image_root),
            "question": row.get("question") or row.get("text") or row.get("query") or "",
            "answer": row.get("answer", row.get("label", "")),
            "category": row.get("category", row.get("type", "")),
            "raw_row": row,
        }
        data.append(item)
        if max_examples > 0 and len(data) >= max_examples:
            break
    return data


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_yes_no(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()

    m = re.search(r"<answer>\s*(yes|no)\s*</answer>", t)
    if m:
        return m.group(1)

    if re.search(r"\b(yes|yeah|yep|true)\b", t):
        return "yes"
    if re.search(r"\b(no|nope|false)\b", t):
        return "no"

    if t in {"1", "y", "t"}:
        return "yes"
    if t in {"0", "n", "f"}:
        return "no"

    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1)

    return t[:16]


def has_answer_tag(text: str) -> bool:
    return bool(re.search(r"<answer>\s*(yes|no)\s*</answer>", text, flags=re.IGNORECASE))


def extract_target_object(question: str) -> str:
    q = question.strip().lower()
    q = re.sub(r"\s+", " ", q)

    patterns = [
        r"is there (?:a|an|any|some) (.+?) in the image\??$",
        r"is there (?:a|an|any|some) (.+?) visible\??$",
        r"is the (.+?) in the image\??$",
        r"are there (.+?) in the image\??$",
        r"do you see (?:a|an|any|some) (.+?) in the image\??$",
    ]

    for pat in patterns:
        m = re.match(pat, q)
        if m:
            obj = m.group(1).strip()
            obj = re.sub(r"[?.!,]+$", "", obj)
            return obj

    return q


def build_main_prompt(item: Dict[str, Any]) -> str:
    target = extract_target_object(item["question"])
    return (
        "Answer the yes/no question about the image.\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Queried object or concept: {target}\n\n"
        "Write one short factual evidence sentence only about the queried object or concept. "
        "Do not mention unrelated objects. "
        "If the queried object is not visible, explicitly say it is not visible.\n"
        "Then end with the exact format <answer>yes</answer> or <answer>no</answer>.\n\n"
        "Format:\n"
        "Evidence: <one short factual sentence about the queried object only>\n"
        "<answer>yes</answer> or <answer>no</answer>"
    )


def build_global_description_prompt(item: Dict[str, Any]) -> str:
    target = extract_target_object(item["question"])
    return (
        f"Question: {item['question']}\n"
        f"Queried object or concept: {target}\n\n"
        "Describe only visual evidence relevant to the queried object or concept. "
        "Do not mention unrelated objects. "
        "If the queried object or concept is not visible, say that it is not visible."
    )


def extract_evidence_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    m = re.search(r"Evidence:\s*(.*?)(?=<answer>|$)", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text.strip()


def enforce_object_consistency(answer: str, evidence: str, target: str) -> str:
    answer = (answer or "").lower().strip()
    evidence = (evidence or "").lower()
    target = (target or "").lower().strip()

    if answer == "yes":
        if target and target not in evidence:
            return "no"
    return answer


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
    max_scan_k: int = 12
    max_evidence_prefixes_per_item: int = 4
    min_evidence_prob: float = 1e-12
    renormalize_after_mix: bool = False
    delta: float = 0.12

    max_decider_calls: int = 4
    decider_prefix_tail_chars: int = 240
    decider_max_new_tokens: int = 40
    decider_model: Optional[str] = None

    max_evidence_items: int = 6
    suppress_direct_answer_letters_first_n_steps: int = 0

    stop_on_eos: bool = True
    ban_eos_at_step0: bool = True
    ban_newline_for_first_n_steps: int = 2

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


def select_candidate_set_knee(probs: torch.Tensor, max_scan_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    kscan = min(max_scan_k + 1, probs.numel())
    sorted_probs, sorted_ids = torch.topk(probs, k=kscan, dim=-1)

    if kscan == 1:
        return sorted_ids[:1], sorted_probs[:1]

    gaps = sorted_probs[:-1] - sorted_probs[1:]
    k_star = int(torch.argmax(gaps).item()) + 1
    k_star = max(1, min(k_star, kscan - 1))
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


def negotiated_reweight_ecrd(
    candidate_probs: torch.Tensor,
    evidence_dist: torch.Tensor,
    renormalize_after_mix: bool = False,
) -> Tuple[torch.Tensor, float]:
    pi_ci = candidate_probs.float().clone()
    ri_ci = evidence_dist.float().clone().to(pi_ci.device)

    pi_mass = pi_ci.sum().clamp_min(1e-12)
    ri_mass = ri_ci.sum().clamp_min(1e-12)
    r_tilde_ci = ri_ci * (pi_mass / ri_mass)

    alpha = float(pi_ci.max().item())
    pmix_ci = alpha * pi_ci + (1.0 - alpha) * r_tilde_ci

    if renormalize_after_mix:
        pmix_ci = safe_normalize(pmix_ci)

    return pmix_ci, alpha


def compute_negotiated_gap(mixed_probs: torch.Tensor) -> float:
    if mixed_probs.numel() <= 1:
        return 1.0
    top2 = torch.topk(mixed_probs, k=2).values
    return float(top2[0].item() - top2[1].item())


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

    def init_evidence_pool(self, image: Image.Image, item: Dict[str, Any]) -> List[EvidenceItem]:
        global_desc = self.base_vlm.generate_text(
            image=image,
            prompt=build_global_description_prompt(item),
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

        probs = safe_normalize(probs)
        candidate_ids, candidate_probs = select_candidate_set_knee(probs=probs, max_scan_k=self.config.max_scan_k)
        return probs, candidate_ids, candidate_probs

    def build_decider_prompt(self, item: Dict[str, Any], prefix_text: str, candidate_tokens: List[str]) -> str:
        tail = truncate_text(prefix_text, self.config.decider_prefix_tail_chars)
        target = extract_target_object(item["question"])
        option_lines = [f"{i}: {repr(tok)}" for i, tok in enumerate(candidate_tokens)]
        option_block = "\n".join(option_lines)

        return (
            "You are a visual decider for yes/no image question answering.\n\n"
            f"Queried object or concept: {target}\n\n"
            "Choose the best NEXT token based only on whether the queried object or concept is visually present.\n"
            "Then provide one short factual visual evidence sentence only about the queried object or concept.\n"
            "Do not mention unrelated objects.\n\n"
            f"Current partial answer:\n{tail}\n\n"
            f"Candidate next tokens:\n{option_block}\n\n"
            "Output format:\n"
            "INDEX: <number>\n"
            "EVIDENCE: <one short factual sentence>"
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
        user_prompt = build_main_prompt(item)

        evidence_pool = self.init_evidence_pool(image=image, item=item)
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
                renormalize_after_mix=self.config.renormalize_after_mix,
            )

            negotiated_gap = compute_negotiated_gap(mixed_probs_t)

            trigger_decider = (
                self.decider_calls < self.config.max_decider_calls
                and len(candidate_token_ids) >= 2
                and negotiated_gap <= self.config.delta
            )

            decider_evidence = ""

            if trigger_decider:
                print(f"[DECIDER] step={step_idx} gap={negotiated_gap:.4f} k={len(candidate_token_ids)}")
                decider_token_id, decider_token, decider_evidence = self.call_visual_decider(
                    image=image,
                    item=item,
                    prefix_text=prefix_text,
                    candidate_token_ids=candidate_token_ids,
                    candidate_tokens=candidate_tokens,
                )
                self.decider_calls += 1

                if decider_token_id is not None and decider_token is not None:
                    self.decider_successes += 1
                else:
                    fallback_idx = int(torch.argmax(mixed_probs_t).item())
                    decider_token_id = candidate_token_ids[fallback_idx]
                    decider_token = candidate_tokens[fallback_idx]

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

            if has_answer_tag(current_text):
                break
            if self.config.stop_on_eos and eos_token_id is not None and chosen_token_id == eos_token_id:
                break

        final_text = self.base_vlm.decode_text(generated_ids).strip()
        evidence_text = extract_evidence_text(final_text)
        pred = normalize_yes_no(final_text)
        pred = enforce_object_consistency(pred, evidence_text, extract_target_object(item["question"]))

        trace = {
            "decider_calls": self.decider_calls,
            "decider_successes": self.decider_successes,
            "num_steps": len(step_infos),
            "num_triggered_steps": sum(int(s.triggered_decider) for s in step_infos),
            "has_answer_tag": has_answer_tag(final_text),
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
    prompt = build_main_prompt(item)
    raw_text = vlm.generate_text(
        image=image,
        prompt=prompt,
        max_new_tokens=64,
        do_sample=False,
    ).strip()

    evidence_text = extract_evidence_text(raw_text)
    pred = normalize_yes_no(raw_text)
    pred = enforce_object_consistency(pred, evidence_text, extract_target_object(item["question"]))

    trace = {
        "has_answer_tag": has_answer_tag(raw_text),
        "num_steps": None,
    }
    return pred, raw_text, trace


def evaluate_single_item(item: Dict[str, Any], mode: str, vlm: VLMWrapper, predictor: Optional[ECRDDecoder]) -> Dict[str, Any]:
    image = open_rgb(item["image_path"])

    if mode == "baseline":
        pred, raw_text, trace = run_baseline(vlm, image, item)
    else:
        predictor.decider_calls = 0
        predictor.decider_successes = 0
        pred, raw_text, trace = predictor.decode(image, item)

    gold = normalize_yes_no(item["answer"])

    out = dict(item)
    out["gold_norm"] = gold
    out["prediction"] = pred
    out["raw_output"] = raw_text
    out["trace"] = trace
    return out


def summarize_results(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(data)
    correct = 0
    yes_total = 0
    no_total = 0
    yes_correct = 0
    no_correct = 0

    for x in data:
        gold = x["gold_norm"]
        pred = x["prediction"]
        if gold == "yes":
            yes_total += 1
            if pred == gold:
                yes_correct += 1
        elif gold == "no":
            no_total += 1
            if pred == gold:
                no_correct += 1
        if pred == gold:
            correct += 1

    return {
        "overall_correct": correct,
        "overall_total": total,
        "overall_acc": (100.0 * correct / total) if total else 0.0,
        "yes_acc": (100.0 * yes_correct / yes_total) if yes_total else 0.0,
        "no_acc": (100.0 * no_correct / no_total) if no_total else 0.0,
        "yes_total": yes_total,
        "no_total": no_total,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"Overall: {summary['overall_correct']}/{summary['overall_total']} = {summary['overall_acc']:.2f}")
    print(f"Yes acc: {summary['yes_acc']:.2f} on {summary['yes_total']}")
    print(f"No acc: {summary['no_acc']:.2f} on {summary['no_total']}")


def parse_args():
    parser = argparse.ArgumentParser(description="POPE baseline / token ECRD")
    parser.add_argument("--mode", type=str, choices=["baseline", "token_ecrd"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--decider_model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--pope_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--delta", type=float, default=0.12)
    parser.add_argument("--max_scan_k", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_decider_calls", type=int, default=4)
    parser.add_argument("--decider_max_new_tokens", type=int, default=40)
    parser.add_argument("--max_evidence_prefixes_per_item", type=int, default=4)
    parser.add_argument("--max_evidence_items", type=int, default=6)

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
            ),
        )

    print("Loading POPE...")
    ds = load_pope(args.pope_path, image_root=args.image_root, max_examples=args.max_examples)

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
        "pope_path": args.pope_path,
        "image_root": args.image_root,
        "summary": summary,
        "results": results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()
