import argparse

from models.qvid_pipeline import (
    add_evaluation_args,
    evaluate_nextqa,
    validate_evaluation_args,
)
from models.qwen3vl.adapter import Qwen3VLBackbone


def build_qwen3vl_caption_prompt(question: str) -> str:
    return f"""
    You are given one sampled frame from a longer video.

    Question: {question}

    Describe the visible scene evidence that could help answer the video question.
    Mention relevant people or animals, objects, locations, poses, hand positions,
    body orientations, and interactions.

    Rules:
    - This is only one frame from a longer video.
    - Do not reject, correct, or deny the premise of the question.
    - Do not conclude that an event never occurs just because it is not visible in this frame.
    - Do not answer the question directly.
    - Even if the target action is unclear, describe the relevant visible context.
    - Write one or two concise factual sentences.
    """


def build_qwen3vl_qa_prompt(captions: str, question: str, options: dict) -> str:
    option_text = "\n".join(
        [f"Option {key}: {value}" for key, value in options.items()]
    )
    valid_letters = ",".join(options.keys())

    prompt = f"""
    Answer the multiple-choice video question using the chronological frame descriptions.

    Some frame descriptions may be incomplete or uncertain.
    Even when the evidence is limited, choose the most plausible option.
    Do not reject the question.
    Do not answer that none of the options can be selected.

    Frame descriptions:
    {captions}

    Question:
    {question}

    Options:
    A. {options["A"]}
    B. {options["B"]}
    C. {options["C"]}
    D. {options["D"]}
    E. {options["E"]}

    Output exactly one uppercase letter: A, B, C, D, or E.
    Do not output any other text.
    """
    return prompt.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_evaluation_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_caption_new_tokens", type=int, default=30)
    parser.add_argument("--max_answer_new_tokens", type=int, default=8)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_evaluation_args(parser, args)

    print("=" * 60)
    print(f"[Evaluation] Requested model: {args.model}")
    print(f"[Evaluation] Device: {args.device or 'auto'}")
    print(f"[Evaluation] Caption max tokens: {args.max_caption_new_tokens}")
    print(f"[Evaluation] Answer max tokens: {args.max_answer_new_tokens}")
    print("=" * 60)

    backbone = Qwen3VLBackbone(
        model_name=args.model,
        device=args.device,
        max_caption_new_tokens=args.max_caption_new_tokens,
        max_answer_new_tokens=args.max_answer_new_tokens,
    )
    evaluate_nextqa(
        backbone=backbone,
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        output=args.output,
        n_frames=args.n_frames,
        limit=args.limit,
        caption_prompt_builder=build_qwen3vl_caption_prompt,
        qa_prompt_builder=build_qwen3vl_qa_prompt,
    )


if __name__ == "__main__":
    main()
