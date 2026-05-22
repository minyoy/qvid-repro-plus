from typing import Dict

import torch
from PIL import Image


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


class Qwen3VLBackbone:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str | None = None,
        max_caption_new_tokens: int = 30,
        max_answer_new_tokens: int = 8,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device or _default_device()
        self.dtype = _default_dtype(self.device)
        self.max_caption_new_tokens = max_caption_new_tokens
        self.max_answer_new_tokens = max_answer_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, messages, max_new_tokens: int, **generation_kwargs) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **generation_kwargs,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], outputs)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_caption(self, image: Image.Image, question: str, prompt: str) -> str:
        del question
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(
            messages,
            max_new_tokens=self.max_caption_new_tokens,
            num_beams=5,
            repetition_penalty=1.5,
        )

    def answer_from_captions(
        self,
        captions: str,
        question: str,
        options: Dict[str, str],
        prompt: str,
    ) -> str:
        del captions, question, options
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        return self._generate(messages, max_new_tokens=self.max_answer_new_tokens)
