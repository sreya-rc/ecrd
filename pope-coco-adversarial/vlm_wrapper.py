import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None

try:
    from transformers import LlavaOnevisionForConditionalGeneration
except Exception:
    LlavaOnevisionForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None


class VLMWrapper:
    """
    Unified wrapper for:
    - llava-hf/llava-1.5-7b-hf
    - llava-hf/llava-onevision-qwen2-7b-ov-hf
    - Qwen/Qwen2.5-VL-3B-Instruct
    - yfan1997/GRIT-20-Qwen2.5-VL-3B
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        if device != "cuda":
            raise ValueError("This wrapper currently expects --device cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.model_name = model_name
        self.device = device
        self.dev = torch.device("cuda:0")
        self.dtype = torch.float16

        lower_name = model_name.lower()
        self.is_onevision = "onevision" in lower_name
        self.is_qwen25vl = (
            "qwen2.5-vl" in lower_name
            or "grit-20-qwen2.5-vl-3b" in lower_name
            or "grit" in lower_name
        )

        if self.is_onevision and LlavaOnevisionForConditionalGeneration is None:
            raise RuntimeError(
                "Your installed transformers does not include LlavaOnevisionForConditionalGeneration."
            )

        if self.is_qwen25vl and Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError(
                "Your installed transformers does not include Qwen2_5_VLForConditionalGeneration. "
                "Upgrade transformers to a recent version."
            )

        if (not self.is_onevision) and (not self.is_qwen25vl) and LlavaForConditionalGeneration is None:
            raise RuntimeError(
                "Your installed transformers does not include LlavaForConditionalGeneration."
            )

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        if self.is_qwen25vl:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif self.is_onevision:
            model_cls = LlavaOnevisionForConditionalGeneration
        else:
            model_cls = LlavaForConditionalGeneration

        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.dev)

        self.model.eval()

    def _build_conversation(self, user_prompt: str, assistant_prefix: str | None = None):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        if assistant_prefix is not None:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_prefix}],
                }
            )
        return conversation

    def _prepare_inputs(
        self,
        image: Image.Image,
        user_prompt: str,
        assistant_prefix: str | None = None,
    ):
        conversation = self._build_conversation(user_prompt, assistant_prefix)

        if assistant_prefix is None:
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )

        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                if v.dtype.is_floating_point:
                    out[k] = v.to(self.dev, dtype=self.dtype)
                else:
                    out[k] = v.to(self.dev)
            else:
                out[k] = v
        return out

    def decode_text(self, token_ids):
        if token_ids is None or len(token_ids) == 0:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    @torch.no_grad()
    def forward_logits_with_assistant_prefix(
        self,
        image: Image.Image,
        user_prompt: str,
        assistant_prefix: str,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(
            image=image,
            user_prompt=user_prompt,
            assistant_prefix=assistant_prefix,
        )
        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :].float()

        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits in forward_logits_with_assistant_prefix")

        return logits

    @torch.no_grad()
    def generate_text(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> str:
        inputs = self._prepare_inputs(
            image=image,
            user_prompt=prompt,
            assistant_prefix=None,
        )

        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = generated[0, prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
