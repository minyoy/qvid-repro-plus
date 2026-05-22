import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Protocol

from PIL import Image

from dataset_parser import load_nextqa
from frame_extractor import extract_frames
from reasoning_module import build_qa_prompt, extract_answer_letter


class QVidBackbone(Protocol):
    def generate_caption(self, image: Image.Image, question: str, prompt: str) -> str:
        ...

    def answer_from_captions(
        self,
        captions: str,
        question: str,
        options: Dict[str, str],
        prompt: str,
    ) -> str:
        ...


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv_path", type=Path, default=Path("data/nextqa/val.csv"))
    parser.add_argument("--video_dir", type=Path, default=Path("videos/nextqa/"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)


def validate_evaluation_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.n_frames <= 0:
        parser.error("--n_frames must be > 0")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")


def build_caption_prompt(question: str) -> str:
    # Keep this text identical to CaptionGenerator.build_prompt.
    return f"Provide a detailed description of the image related to the {question}"


def generate_captions_for_frames(
    backbone: QVidBackbone,
    frames: List[Image.Image],
    question: str,
) -> List[Dict[str, Any]]:
    caption_results = []
    prompt = build_caption_prompt(question)

    for frame_idx, frame in enumerate(frames):
        if not isinstance(frame, Image.Image):
            raise TypeError(f"unsupported frame type: {type(frame)}")

        caption = backbone.generate_caption(
            image=frame.convert("RGB"),
            question=question,
            prompt=prompt,
        )
        caption_results.append({"frame_idx": frame_idx, "caption": caption})

    return caption_results


def aggregate_captions(caption_results: List[Dict[str, Any]]) -> str:
    return " ".join(item["caption"] for item in caption_results)


def save_json(rows: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file_obj:
        json.dump(rows, file_obj, ensure_ascii=False, indent=2)


def evaluate_nextqa(
    backbone: QVidBackbone,
    csv_path: Path,
    video_dir: Path,
    output: Path,
    n_frames: int,
    limit: int | None = None,
) -> Dict[str, Any]:
    samples = load_nextqa(csv_path, video_dir)
    if limit is not None:
        samples = samples[:limit]

    results = []
    correct = 0
    total = 0

    for idx, sample in enumerate(samples):
        print(f"\n[{idx + 1}/{len(samples)}] video_id={sample['video_id']}")
        start_time = time.time()

        try:
            frames = extract_frames(sample["video_path"], n_frames=n_frames)
            caption_results = generate_captions_for_frames(
                backbone=backbone,
                frames=frames,
                question=sample["question"],
            )
            captions = aggregate_captions(caption_results)
            qa_prompt = build_qa_prompt(
                captions=captions,
                question=sample["question"],
                options=sample["options"],
            )
            raw_output = backbone.answer_from_captions(
                captions=captions,
                question=sample["question"],
                options=sample["options"],
                prompt=qa_prompt,
            )
            pred = extract_answer_letter(raw_output, sample["options"].keys())

            gold = sample["answer"]
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1

            result = {
                "video_id": sample["video_id"],
                "question": sample["question"],
                "options": sample["options"],
                "gold": gold,
                "pred": pred,
                "raw_output": raw_output,
                "correct": is_correct,
                "n_frames": n_frames,
                "elapsed_sec": time.time() - start_time,
                "captions": captions,
            }
            print(
                f"gold={gold}, "
                f"pred={pred}, "
                f"raw={raw_output}, "
                f"correct={is_correct}"
            )
        except Exception as exc:
            total += 1
            result = {
                "video_id": sample.get("video_id"),
                "error": str(exc),
                "correct": False,
                "n_frames": n_frames,
            }
            print(f"ERROR: {exc}")

        results.append(result)

    accuracy = correct / total if total > 0 else 0.0
    payload = {
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,
            "n_frames": n_frames,
        },
        "results": results,
    }
    save_json(payload, output)

    print("\n=== Evaluation Summary ===")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Saved to: {output}")
    return payload
