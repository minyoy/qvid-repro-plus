import argparse

from models.internvl35.adapter import InternVL35Backbone
from models.qvid_pipeline import (
    add_evaluation_args,
    evaluate_nextqa,
    validate_evaluation_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_evaluation_args(parser)
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_caption_new_tokens", type=int, default=30)
    parser.add_argument("--max_answer_new_tokens", type=int, default=8)
    parser.add_argument("--max_num_tiles", type=int, default=12)
    parser.add_argument("--use_flash_attn", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_evaluation_args(parser, args)

    backbone = InternVL35Backbone(
        model_name=args.model,
        device=args.device,
        max_caption_new_tokens=args.max_caption_new_tokens,
        max_answer_new_tokens=args.max_answer_new_tokens,
        max_num_tiles=args.max_num_tiles,
        use_flash_attn=args.use_flash_attn,
    )
    evaluate_nextqa(
        backbone=backbone,
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        output=args.output,
        n_frames=args.n_frames,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
