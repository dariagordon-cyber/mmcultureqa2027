import re
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# 1. Configuration
# MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_ID = "google/gemma-4-31B-it"

QUANTIZATION = "bitsandbytes" 
LOAD_FORMAT = "bitsandbytes"

THINKING_MODE = False # Toggle thinking on or off

if __name__ == "__main__":
    # 2. Initialize Tokenizer
    tokenizer = AutoProcessor.from_pretrained(MODEL_ID)

    # 3. Initialize vLLM
    print(f"Loading {MODEL_ID} in 4-bit...")
    llm = LLM(
        model=MODEL_ID,
        quantization=QUANTIZATION,
        load_format=LOAD_FORMAT,
        enforce_eager=True, 
        gpu_memory_utilization=0.90,
        max_model_len=4096, 
        trust_remote_code=True
    )

    # --- USER HELPER FUNCTIONS ---

    def clean_thinking_output(text: str) -> str:
        """Extract only the final answer from model output."""
        if not text:
            return ""

        if "FINAL_ANSWER:" in text:
            return text.rsplit("FINAL_ANSWER:", 1)[-1].strip()

        if "<channel|>" in text:
            return text.rsplit("<channel|>", 1)[-1].strip()

        if "</think>" in text:
            return text.split("</think>", 1)[-1].strip()

        return text.strip()

    def apply_chat_template_safe(tokenizer, messages, enable_thinking=False):
        """Applies chat template, passing enable_thinking if supported by tokenizer."""    
        try:
            # Works for models like Qwen that support the custom kwarg
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        except TypeError:
            # Fallback for models like Gemma that do not support the kwarg
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def build_messages(question: str, use_thinking: bool):
        """Builds the message dictionary for multi-modal standard format."""
        messages = []
    
        # We still include the system prompt for models that fallback in apply_chat_template_safe
        if use_thinking:
            messages.append({
                "role": "system", 
                "content": "You are an expert visual AI assistant. Think step-by-step about the image before answering. At the very end, on a new line, write FINAL_ANSWER: followed by only the short final answer."
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a direct and concise visual AI assistant. Answer the question directly without outputting your reasoning steps. Write FINAL_ANSWER: followed by only the short final answer."
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        })
    
        return messages

    # --- INFERENCE PIPELINE ---

    image_path = "sample-item-2.jpg"
    image = Image.open(image_path).convert("RGB")
    question = "What is the name of the building shown in the image, and what inspired its design?"

    # Build messages and apply template using the safe function
    messages = build_messages(question, THINKING_MODE)
    formatted_prompt = apply_chat_template_safe(
        tokenizer, 
        messages, 
        enable_thinking=THINKING_MODE
    )

    # 4. Set Sampling Parameters (Updated lengths based on request)
    if "Qwen" in MODEL_ID:
        sampling_params = SamplingParams(
            temperature=1.0 if THINKING_MODE else 0.7,
            top_p=0.95 if THINKING_MODE else 0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0 if THINKING_MODE else 1.5,
            repetition_penalty=1.0,
            max_tokens=3064 if THINKING_MODE else 1024,
            stop=["<|im_end|>", "<|eot_id|>"]
        )
    else:
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=3064 if THINKING_MODE else 1024,
            skip_special_tokens=False,
            stop=["<|im_end|>", "<|eot_id|>"]
        )

    # 5. Run Inference
    print("Running inference...")
    outputs = llm.generate(
        {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params
    )

    # 6. Output Parsing
    raw_text = outputs[0].outputs[0].text
    cleaned_text = clean_thinking_output(raw_text)

    print("\n=== RAW OUTPUT (Includes Thoughts) ===")
    print(raw_text)

    print("\n=== CLEANED FINAL ANSWER ===")
    print(cleaned_text)
