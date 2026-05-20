import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import List, Dict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class CaptionGenerator:
    def __init__(
        self,
        model_name: str = "Salesforce/instructblip-flan-t5-xl",
        device: str = None,
        max_new_tokens: int = 256,
    ):
        self.device = device or get_device()
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        self.processor = InstructBlipProcessor.from_pretrained(model_name)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    # Q-ViD 핵심: 일반 캡션 대신 질문을 프롬프트에 포함시켜
    # 질문과 관련된 내용만 집중적으로 캡션 생성

    def build_prompt(self, question: str) -> str:
        return f"Provide a detailed description of the image related to the {question}"

    @torch.no_grad()
    def generate_caption(self, image: Image.Image, question: str) -> str:
        prompt = self.build_prompt(question)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=5,
            repetition_penalty=1.5,
        )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return caption

    # 반환 형식: [{"frame_idx": 0, "caption": "..."}, {"frame_idx": 1, "caption": "..."}, ...]
    def generate_captions_for_frames(
        self,
        frames: List,
        question: str,
        verbose: bool = True,
    ) -> List[Dict]:
        if not frames:
            logger.warning("프레임이 없습니다.")
            return []

        results = []
        for idx, frame in enumerate(frames):
            if verbose:
                logger.info(f"프레임 [{idx + 1}/{len(frames)}] 처리 중...")

            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame).convert("RGB")
            elif isinstance(frame, Image.Image):
                if frame.mode != "RGB":
                    frame = frame.convert("RGB")
            else:
                raise TypeError(f"지원하지 않는 프레임 형식: {type(frame)}")

            caption = self.generate_caption(frame, question)
            results.append({"frame_idx": idx, "caption": caption})

            if verbose:
                preview = caption[:80] + ("..." if len(caption) > 80 else "")
                logger.info(f"캡션: {preview}")

        return results

    # 프레임별 캡션을 시간순으로 이어 붙인 문자열 반환
    # 이 문자열이 Flan-T5 QA 프롬프트의 captions 파트로 사용됨
    def aggregate_captions(self, caption_results: List[Dict]) -> str:
        captions = [item["caption"] for item in caption_results]
        return " ".join(captions)

# from caption_generator import generate_captions
# aggregated_captions = generate_captions(frames, question)
# aggregated_captions를 Flan-T5 프롬프트에 넣으면 됨
def generate_captions(
    frames: List,
    question: str,
    model_name: str = "Salesforce/instructblip-flan-t5-xl",
    device: str = None,
) -> str:
    generator = CaptionGenerator(model_name=model_name, device=device)
    caption_results = generator.generate_captions_for_frames(frames, question)
    return generator.aggregate_captions(caption_results)


if __name__ == "__main__":
    generator = CaptionGenerator()

    dummy_image = Image.new("RGB", (224, 224), color=(128, 64, 32))
    question = "Why did the man move the gift to the sofa?"

    dummy_frames = [dummy_image] * 3
    results = generator.generate_captions_for_frames(dummy_frames, question)

    for r in results:
        print(f"Frame {r['frame_idx']}: {r['caption']}")

    print(generator.aggregate_captions(results))