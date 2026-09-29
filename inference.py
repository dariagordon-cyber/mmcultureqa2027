import re
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 1. Configuration
# MODEL_ID = "Qwen/Qwen3.8-27B" 
MODEL_ID = "google/gemma-4-31B-it" 

QUANTIZATION = "bitsandbytes" 
LOAD_FORMAT = "bitsandbytes"

THINKING_MODE = True # Toggle thinking on or off

# 2. Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
    """Removes thinking traces (<think>...</think>) and leaves only the final output."""    
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    else:
        # Handles cases where max_tokens was reached before closing tag
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
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
            "content": "You are an expert visual AI assistant. Please think step-by-step to reason about the image before answering. Enclose your reasoning inside <think> and </think> tags, then provide your final answer."
        })
    else:
        messages.append({
            "role": "system",
            "content": "You are a direct and concise visual AI assistant. Answer the question directly without outputting your reasoning steps."
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

image_path = "path/to/your/image.jpg"
image = Image.open(image_path).convert("RGB")
question = "What is the primary action happening in this image?"

# Build messages and apply template using the safe function
messages = build_messages(question, THINKING_MODE)
formatted_prompt = apply_chat_template_safe(
    tokenizer, 
    messages, 
    enable_thinking=THINKING_MODE
)

# 4. Set Sampling Parameters (Updated lengths based on request)
sampling_params = SamplingParams(
    temperature=0.6 if THINKING_MODE else 0.2, 
    max_tokens=3064 if THINKING_MODE else 1024,
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
