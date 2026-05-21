import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def build_qa_prompt(captions: str, question: str, options: dict) -> str:
    option_text = "\n".join(
        [f"Option {key}: {value}" for key, value in options.items()]
    )
    valid_letters = ",".join(options.keys())

    prompt = f"""Captions:
{captions}

Question:
{question}

{option_text}

Considering the information presented in the captions, select the correct answer in one letter from the options ({valid_letters}).
"""
    return prompt.strip()


def extract_answer_letter(text: str, valid_options) -> str | None:
    valid_options = set([str(x).upper() for x in valid_options])
    text = text.strip().upper()

    if text in valid_options:
        return text

    match = re.search(r"\b([A-E])\b", text)
    if match:
        letter = match.group(1)
        if letter in valid_options:
            return letter

    return None


class ReasoningModule:
    def __init__(self, model_name: str = "google/flan-t5-base", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_raw(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def predict(self, captions: str, question: str, options: dict):
        prompt = build_qa_prompt(captions, question, options)
        raw_output = self.predict_raw(prompt)
        pred = extract_answer_letter(raw_output, options.keys())
        return pred, raw_output, prompt