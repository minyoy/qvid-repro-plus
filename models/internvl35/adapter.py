from typing import Dict

import torch
from PIL import Image

from models.internvl35.image_processing import image_to_pixel_values


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


class InternVL35Backbone:
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-4B",
        device: str | None = None,
        max_caption_new_tokens: int = 30,
        max_answer_new_tokens: int = 8,
        max_num_tiles: int = 12,
        use_flash_attn: bool = False,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or _default_device()
        self.dtype = _default_dtype(self.device)
        self.max_caption_new_tokens = max_caption_new_tokens
        self.max_answer_new_tokens = max_answer_new_tokens
        self.max_num_tiles = max_num_tiles

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    @torch.inference_mode()
    def generate_caption(self, image: Image.Image, question: str, prompt: str) -> str:
        del question
        pixel_values = image_to_pixel_values(
            image=image,
            max_num_tiles=self.max_num_tiles,
        ).to(device=self.device, dtype=self.dtype)
        generation_config = {
            "max_new_tokens": self.max_caption_new_tokens,
            "do_sample": False,
            "num_beams": 5,
            "repetition_penalty": 1.5,
        }
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            f"<image>\n{prompt}",
            generation_config,
        ).strip()

    @torch.inference_mode()
    def answer_from_captions(
        self,
        captions: str,
        question: str,
        options: Dict[str, str],
        prompt: str,
    ) -> str:
        del captions, question, options
        generation_config = {
            "max_new_tokens": self.max_answer_new_tokens,
            "do_sample": False,
        }
        return self.model.chat(
            self.tokenizer,
            None,
            prompt,
            generation_config,
        ).strip()
