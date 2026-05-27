"""
Task 1: Sentiment labeling with an LLM.

Why an LLM here?
The messages in this dataset are described as scattered, partly non-work-related,
and nuanced. Rule-based lexicons such as VADER are strong for short, clearly
polarity-driven text, but they often miss context, sarcasm, indirect tone,
social chatter, and mixed-intent emails. A Large Language Model can evaluate the
full message context and produce a more faithful Positive / Negative / Neutral
label.

Output:
    labeled_data.csv with a new Sentiment_Label column.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from email_analysis_utils import combine_text_columns, load_csv, pretty_print_section, resolve_column_name

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import error is environment-specific
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai is optional now
    OpenAI = None


VALID_LABELS = {"Positive", "Negative", "Neutral"}
LOCAL_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class LocalLLMClassifier:
    """Lazy-loaded local transformer classifier."""

    model_name: str
    tokenizer: object
    model: object


def build_local_classifier(model_name: str) -> LocalLLMClassifier:
    """
    Load a local Hugging Face causal LM and tokenizer.

    This uses the exact chat-template style you provided. The default model is a
    smaller 0.5B instruction-tuned Qwen model that is more practical on a normal
    laptop than the original 20B option.
    """

    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "transformers/torch are required for local inference. Install them with "
            "`pip install transformers torch accelerate`."
        ) from _TRANSFORMERS_IMPORT_ERROR

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return LocalLLMClassifier(model_name=model_name, tokenizer=tokenizer, model=model)


def build_client() -> "OpenAI":
    """Create an OpenAI client using the standard OPENAI_API_KEY environment variable."""

    if OpenAI is None:
        raise ImportError("openai package is not installed.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add your OpenAI API key to the environment "
            "before running task1_sentiment_labeling.py."
        )
    return OpenAI(api_key=api_key)


def classify_sentiment_with_openai(
    client: "OpenAI",
    message: str,
    model: str,
    max_retries: int = 5,
    base_delay_seconds: float = 1.0,
) -> str:
    """
    Send one message to the LLM and return a cleaned sentiment label.

    The prompt intentionally requests a single-word response so the output stays
    machine-readable.
    """

    system_prompt = (
        "You are a strict sentiment classifier for employee messages. Carefully "
        "evaluate the full context, tone, subtle cues, sarcasm, politeness, "
        "frustration, gratitude, and any non-work-related content if present. "
        "Return ONLY one word: Positive, Negative, or Neutral. Do not explain."
    )

    user_prompt = message.strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = response.choices[0].message.content or ""
            cleaned = raw_text.strip().split()[0].capitalize()
            if cleaned not in VALID_LABELS:
                raise ValueError(f"Unexpected label returned by model: {raw_text!r}")
            return cleaned
        except Exception as exc:
            if attempt == max_retries - 1:
                print(
                    f"Warning: failed to classify message after {max_retries} attempts. "
                    f"Falling back to Neutral. Error: {exc}"
                )
                return "Neutral"
            sleep_for = base_delay_seconds * (2 ** attempt)
            print(
                f"API call failed on attempt {attempt + 1}/{max_retries}. "
                f"Retrying in {sleep_for:.1f}s. Error: {exc}"
            )
            time.sleep(sleep_for)


def classify_sentiment_with_local_model(
    classifier: LocalLLMClassifier,
    message: str,
    max_new_tokens: int = 12,
) -> str:
    """
    Classify one message with a local Hugging Face causal LM.

    The message is wrapped in a chat template, generated, then the first token
    of the response is normalized to Positive / Negative / Neutral.
    """

    system_prompt = (
        "You are a strict sentiment classifier for employee messages. Carefully "
        "evaluate the full context, tone, subtle cues, sarcasm, politeness, "
        "frustration, gratitude, and any non-work-related content if present. "
        "Return ONLY one word: Positive, Negative, or Neutral. Do not explain."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message.strip()},
    ]

    inputs = classifier.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(classifier.model.device)

    with torch.no_grad():
        outputs = classifier.model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded = classifier.tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    cleaned = decoded.split()[0].capitalize() if decoded else "Neutral"
    if cleaned not in VALID_LABELS:
        return "Neutral"
    return cleaned


def label_messages(
    df: pd.DataFrame,
    subject_column: str,
    body_column: str,
    provider: str,
    local_model_name: str,
    openai_model_name: str,
    delay_seconds: float,
    max_retries: int,
) -> pd.Series:
    """
    Label each message while caching duplicate messages so the same text is not
    sent to the API more than once.

    To run locally without an API key, this script defaults to the Hugging Face
    transformer path using `Qwen/Qwen2.5-0.5B-Instruct`. If you prefer another
    local chat model, set `HF_MODEL_NAME` and keep the same chat-template pattern.
    """

    cache: Dict[str, str] = {}
    labels = []
    local_classifier: Optional[LocalLLMClassifier] = None
    client: Optional["OpenAI"] = None

    if provider in {"auto", "local"}:
        try:
            local_classifier = build_local_classifier(local_model_name)
            print(f"Loaded local transformer model: {local_model_name}")
        except Exception as exc:
            if provider == "local":
                raise
            print(f"Local model load failed, falling back to OpenAI if available. Error: {exc}")
            client = build_client()
            print(f"Using OpenAI model: {openai_model_name}")
    elif provider == "openai":
        client = build_client()
        print(f"Using OpenAI model: {openai_model_name}")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    combined_text = combine_text_columns(df[subject_column], df[body_column])

    for idx, message in enumerate(combined_text):
        message = str(message).strip()
        if not message:
            labels.append("Neutral")
            continue

        if message not in cache:
            if provider in {"auto", "local"}:
                try:
                    cache[message] = classify_sentiment_with_local_model(
                        classifier=local_classifier,
                        message=message,
                    )
                except Exception as exc:
                    if provider == "local":
                        print(
                            f"Local model failed to classify a message. "
                            f"Falling back to Neutral. Error: {exc}"
                        )
                        cache[message] = "Neutral"
                    else:
                        print(
                            f"Local model failed, trying OpenAI fallback if available. "
                            f"Error: {exc}"
                        )
                        client = build_client()
                        cache[message] = classify_sentiment_with_openai(
                            client=client,
                            message=message,
                            model=openai_model_name,
                            max_retries=max_retries,
                        )
            else:
                cache[message] = classify_sentiment_with_openai(
                    client=client,
                    message=message,
                    model=openai_model_name,
                    max_retries=max_retries,
                )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        labels.append(cache[message])

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1} messages...")

    return pd.Series(labels, index=df.index, name="Sentiment_Label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label employee messages with sentiment.")
    parser.add_argument("--input", default="test(in).csv", help="Input CSV path.")
    parser.add_argument("--output", default="labeled_data.csv", help="Output CSV path.")
    parser.add_argument(
        "--subject-column",
        default=None,
        help="Subject column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--body-column",
        default=None,
        help="Body column name, if not one of the default candidates.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "local", "openai"],
        default="auto",
        help="Choose local transformers, OpenAI API, or auto fallback.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Deprecated alias for --local-model. Use --local-model or --openai-model instead.",
    )
    parser.add_argument(
        "--local-model",
        default=LOCAL_MODEL_NAME,
        help="Local Hugging Face model name. Defaults to Qwen/Qwen2.5-0.5B-Instruct.",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_MODEL_NAME,
        help="OpenAI model name used only when provider=openai or auto falls back to OpenAI.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.1,
        help="Delay between successful API calls to reduce burst rate.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for transient API failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretty_print_section("Task 1: Sentiment Labeling")

    df = load_csv(args.input)
    subject_column = resolve_column_name(
        df,
        args.subject_column,
        ["subject", "Subject", "SUBJECT"],
        "subject",
    )
    body_column = resolve_column_name(
        df,
        args.body_column,
        ["body", "Body", "BODY", "message", "Message"],
        "body",
    )

    print(f"Using subject column: {subject_column}")
    print(f"Using body column: {body_column}")
    print(f"Using provider: {args.provider}")
    print(f"Using local model: {args.local_model}")
    print(f"Using OpenAI model: {args.openai_model}")

    if args.model:
        print("Warning: --model is deprecated. Use --local-model and --openai-model instead.")
        args.local_model = args.model
        args.openai_model = args.model

    df["Sentiment_Label"] = label_messages(
        df=df,
        subject_column=subject_column,
        body_column=body_column,
        provider=args.provider,
        local_model_name=args.local_model,
        openai_model_name=args.openai_model,
        delay_seconds=args.delay_seconds,
        max_retries=args.max_retries,
    )

    output_path = args.output
    df.to_csv(output_path, index=False)
    print(f"Saved labeled dataset to: {output_path}")


if __name__ == "__main__":
    main()
