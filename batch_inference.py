import argparse
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODELS = {
    "qwen": "Qwen/Qwen3.8-27B",
    "gemma": "google/gemma-4-31B-it",
}


def str_to_bool(value):
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def clean_output(text: str) -> str:
    """Return only the final answer, without reasoning."""
    if not text:
        return ""

    if "FINAL_ANSWER:" in text:
        return text.rsplit("FINAL_ANSWER:", 1)[-1].strip()

    # Gemma thinking format
    if "<channel|>" in text:
        return text.rsplit("<channel|>", 1)[-1].strip()

    # Qwen-style thinking format
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()

    return text.strip()


def apply_chat_template(processor, messages, thinking):
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_messages(question):
    # Keep the task instruction identical across thinking/non-thinking modes.
    return [
        {
            "role": "system",
            "content": (
                "Answer the question based on the image. "
                "Give a short, direct answer. Prefer one sentence and use at most two short sentences."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def get_sampling_params(model_name, thinking):
    if model_name == "qwen":
        return SamplingParams(
            temperature=1.0 if thinking else 0.7,
            top_p=0.95 if thinking else 0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0 if thinking else 1.5,
            repetition_penalty=1.0,
            max_tokens=3064 if thinking else 1024,
            stop=["<|im_end|>", "<|eot_id|>"],
        )

    return SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=3064 if thinking else 1024,
        skip_special_tokens=False,
        stop=["<|im_end|>", "<|eot_id|>"],
    )


def load_completed_ids(output_path):
    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("id"):
                    completed.add(row["id"])
            except json.JSONDecodeError:
                pass

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "gemma"], required=True)
    parser.add_argument("--thinking", type=str_to_bool, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    model_id = MODELS[args.model]
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_ids(output_path)

    print(f"Model: {model_id}")
    print(f"Thinking: {args.thinking}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Already completed: {len(completed_ids)}")

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading {model_id} in 4-bit...")
    llm = LLM(
        model=model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    sampling_params = get_sampling_params(args.model, args.thinking)

    processed = 0

    with input_path.open("r", encoding="utf-8") as input_file, \
         output_path.open("a", encoding="utf-8") as output_file:

        for line in input_file:
            item = json.loads(line)
            example_id = item["id"]

            if example_id in completed_ids:
                continue

            if args.limit is not None and processed >= args.limit:
                break

            image_path = input_path.parent / item["image"]

            try:
                image = Image.open(image_path).convert("RGB")

                messages = build_messages(item["question"])
                prompt = apply_chat_template(
                    processor,
                    messages,
                    args.thinking,
                )

                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    },
                    sampling_params=sampling_params,
                )

                raw_text = outputs[0].outputs[0].text
                prediction = clean_output(raw_text)

                result = {
                    "id": example_id,
                    "image": item["image"],
                    "country": item.get("country"),
                    "category": item.get("category"),
                    "subcategory": item.get("subcategory"),
                    "question": item["question"],
                    "reference": item.get("answer"),
                    "model": model_id,
                    "thinking": args.thinking,
                    "prediction": prediction,
                }

                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                processed += 1
                print(
                    f"[{processed}] {example_id[:10]}... "
                    f"Prediction: {prediction}"
                )

            except Exception as exc:
                print(f"ERROR on {example_id}: {exc}")

    print(f"Finished. New examples processed: {processed}")


if __name__ == "__main__":
    main()
