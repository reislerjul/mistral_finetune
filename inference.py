from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from constants import MAX_TOKEN_RESPONSE_LENGTH, MAX_TOKEN_CONTEXT_LENGTH, MODEL_NAME

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="balanced_low_0")

# CHAI formatting templates
formatter = {
    "memory_template": "{bot_name}'s Persona: {memory}\n####\n",
    "prompt_template": "{prompt}\n<START>\n",
    "bot_template": "{bot_name}: {message}\n",
    "user_template": "{user_name}: {message}\n",
    "response_template": "{bot_name}:",
    "truncate_by_message": False
}

def format_chai_prompt(memory, history, formatter, bot_name="Nova", user_name="you", example_history=None, tokenizer=None, max_context_tokens=MAX_TOKEN_CONTEXT_LENGTH):
    """
    Build a CHAI-style prompt with memory, optional example history, and live conversation.
    Truncates example history first, then chat history, keeping memory intact.
    """
    # Step 1: Memory (always included)
    memory_block = formatter["memory_template"].format(bot_name=bot_name, memory=memory)
    
    # Step 2: Example history (truncated first)
    example_block = ""
    if example_history:
        for msg in example_history:
            if msg["role"] == "user":
                example_block += formatter["user_template"].format(user_name=user_name, message=msg["content"])
            elif msg["role"] == "assistant":
                example_block += formatter["bot_template"].format(bot_name=bot_name, message=msg["content"])
        example_block = formatter["prompt_template"].format(prompt=example_block)

    # Step 3: Chat history (truncated second)
    chat_block = ""
    for msg in history:
        if msg["role"] == "user":
            chat_block += formatter["user_template"].format(user_name=user_name, message=msg["content"])
        elif msg["role"] == "assistant":
            chat_block += formatter["bot_template"].format(bot_name=bot_name, message=msg["content"])

    # Combine prompt + response prefix
    prompt = memory_block + example_block + chat_block
    prompt += formatter["response_template"].format(bot_name=bot_name)

    # Step 4: Truncate by tokens if needed
    if tokenizer is not None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        if len(input_ids) > max_context_tokens:
            # Truncate example history first
            if example_history:
                return format_chai_prompt(memory, history, formatter, bot_name, user_name, example_history[:-1], tokenizer, max_context_tokens)
            # Then truncate chat history from the top
            if history:
                return format_chai_prompt(memory, history[1:], formatter, bot_name, user_name, example_history, tokenizer, max_context_tokens)

    return prompt

# === Example usage ===
memory = "Nova is a sarcastic but empathetic AI who enjoys riddles and hates small talk."

# Optional example dialogue that sets expectations (this gets truncated first)
example_history = [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I'm Nova, your sardonic digital assistant."}
]

# Live chat history
chat_history = [
    {"role": "user", "content": "Hey Nova."},
    {"role": "assistant", "content": "Oh good, you're back. I was just enjoying the silence."},
    {"role": "user", "content": "What's your opinion on small talk?"}
]

# Build the full prompt
prompt = format_chai_prompt(
    memory=memory,
    history=chat_history,
    example_history=example_history,
    formatter=formatter,
    bot_name="Nova",
    user_name="you",
    tokenizer=tokenizer,
    max_context_tokens=1024  # can increase if needed
)

# Tokenize and generate
print("=== Generated Prompt ===")
print(prompt.strip())

#input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(
        input_ids=model_input["input_ids"],
        attention_mask=model_input["attention_mask"],
        max_new_tokens=MAX_TOKEN_RESPONSE_LENGTH,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode only new tokens
print("\n=== Nova's Response ===")
decoded = tokenizer.decode(output[0][model_input["input_ids"].shape[-1]:], skip_special_tokens=True)
response = decoded.strip().split("\n")[0]
print(f"Nova: {response.strip()}")
