"""
evaluate.py

Q-ViD의 전체 파이프라인 실행 파일입니다.

데이터셋 로드, 비디오 프레임 추출, 질문 기반 프레임 캡션 생성,
LLM 기반 정답 예측, accuracy 계산 및 결과 저장까지의 과정을 하나로 연결합니다.

즉, 다음 흐름을 실행하는 엔트리포인트입니다.
    CSV annotation
    → Video frame extraction
    → Question-dependent frame captioning
    → LLM reasoning
    → Evaluation

주요 입력:
- NExT-QA 형식의 CSV 파일
- 비디오 파일들이 저장된 디렉터리
- 사용할 프레임 수
- 평가할 샘플 수

주요 출력:
- 샘플별 예측 결과
- 전체 accuracy summary
- JSON 결과 파일
"""

import argparse
import json
import time
from pathlib import Path

from dataset_parser import load_nextqa
from frame_extractor import extract_frames
from caption_generator import CaptionGenerator
from reasoning_module import ReasoningModule

def save_json(rows, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, default=Path("data/nextqa/val.csv"))
    parser.add_argument("--video_dir", type=Path, default=Path("videos/nextqa/"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)

    # 프레임 캡션 생성을 위한 InstructBLIP 모델
    # 논문 메인 실험은 InstructBLIP-Flan-T5XXL을 사용했지만,
    # 로컬 개발 환경의 메모리 제약을 고려하여 XL 모델을 기본값으로 사용한다.
    parser.add_argument("--caption_model", default="Salesforce/instructblip-flan-t5-xl")

    # 캡션, 질문, 선택지를 보고 최종 정답을 고르는 reasoning 모델
    # 논문에서는 Flan-T5XXL을 사용했지만,
    # 맥북/로컬 환경에서 빠른 테스트가 가능하도록 Flan-T5-base를 기본값으로 사용한다.
    parser.add_argument("--reasoning_model", default="google/flan-t5-base")

    args = parser.parse_args()

    if args.n_frames <= 0:
        parser.error("--n_frames must be > 0")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples = load_nextqa(args.csv_path, args.video_dir)

    if args.limit is not None:
        samples = samples[:args.limit]

    caption_generator = CaptionGenerator(model_name=args.caption_model)
    reasoner = ReasoningModule(model_name=args.reasoning_model)

    results = []
    correct = 0
    total = 0

    for idx, sample in enumerate(samples):
        print(f"\n[{idx + 1}/{len(samples)}] video_id={sample['video_id']}")

        start_time = time.time()

        try:
            frames = extract_frames(sample["video_path"], n_frames=args.n_frames)

            caption_results = caption_generator.generate_captions_for_frames(
                frames=frames,
                question=sample["question"],
                verbose=False,
            )
            captions = caption_generator.aggregate_captions(caption_results)

            pred, raw_output, prompt = reasoner.predict(
                captions=captions,
                question=sample["question"],
                options=sample["options"],
            )

            gold = sample["answer"]
            is_correct = pred == gold

            if is_correct:
                correct += 1
            total += 1

            elapsed = time.time() - start_time

            result = {
                "video_id": sample["video_id"],
                "question": sample["question"],
                "options": sample["options"],
                "gold": gold,
                "pred": pred,
                "raw_output": raw_output,
                "correct": is_correct,
                "n_frames": args.n_frames,
                "elapsed_sec": elapsed,         # 샘플 처리 시간
                "captions": captions,           # InstructBLIP이 생성한 프레임 캡션들
            }

            print(
                f"gold={gold}, "  # 실제 정답
                f"pred={pred}, "  # 모델 예측 정답
                f"raw={raw_output}, "  # 모델 원문 출력
                f"correct={is_correct}"  # 정답 여부
            )
        except Exception as e:
            total += 1
            result = {
                "video_id": sample.get("video_id"),
                "error": str(e),
                "correct": False,
                "n_frames": args.n_frames,
            }
            print(f"ERROR: {e}")

        results.append(result)

    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "total": total,  # 평가한 전체 샘플 수
        "correct": correct,  # 모델이 맞힌 샘플 수
        "accuracy": accuracy,  # 정확도, 0~1 사이 값
        "accuracy_percent": accuracy * 100,  # 정확도, 퍼센트 값
        "n_frames": args.n_frames,  # 샘플당 사용한 프레임 수
    }

    save_json(
        {
            "summary": summary,
            "results": results,
        },
        args.output,
    )

    print("\n=== Evaluation Summary ===")
    print(f"Total: {total}")  # 평가한 전체 샘플 수
    print(f"Correct: {correct}")  # 맞힌 샘플 수
    print(f"Accuracy: {accuracy * 100:.2f}%")  # 최종 정확도
    print(f"Saved to: {args.output}")  # 결과가 저장된 JSON 파일 경로


if __name__ == "__main__":
    main()
