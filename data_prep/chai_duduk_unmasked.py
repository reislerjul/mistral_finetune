import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from constants import MODEL_NAME, MAX_TOKEN_CONTEXT_LENGTH


def process_sample(sample, tokenizer):
    memory = sample["memory"]
    conversation = sample["conversation"]
    bot_label = sample["bot_label"]

    # Replace '|newline|' with actual newlines
    conversation = conversation.replace("|newline|", "\n")

    # Construct prompt preamble
    preamble = f"{bot_label}'s Persona: {memory}\n####\n"

    # Tokenize preamble
    preamble_tokens = tokenizer(preamble, add_special_tokens=False)
    preamble_input_ids = preamble_tokens["input_ids"]
    preamble_len = len(preamble_input_ids)

    # Split conversation into turns
    turns = conversation.split("\n")

    # Tokenize each turn and track which are bot vs user
    turns_with_info = []
    for i, turn in enumerate(turns):
        if i == 0 and turn.startswith(f"{bot_label}:"):
            include_in_labels = False  # first bot turn
            text = turn
        elif turn.startswith(f"{bot_label}:"):
            include_in_labels = True
            text = turn
        else:
            include_in_labels = True
            text = turn

        tokens = tokenizer(text + "\n", add_special_tokens=False)
        turns_with_info.append((tokens["input_ids"], include_in_labels))

    # Assemble tokenized chunks <= MAX_TOKEN_CONTEXT_LENGTH
    chunks = []
    current_input_ids = preamble_input_ids[:]
    current_labels = [-100] * preamble_len

    for tokens, include_in_labels in turns_with_info:
        if len(current_input_ids) + len(tokens) > MAX_TOKEN_CONTEXT_LENGTH:
            # Save current chunk
            chunks.append({
                "input_ids": current_input_ids,
                "labels": current_labels,
            })
            # Start new chunk with preamble
            current_input_ids = preamble_input_ids[:]
            current_labels = [-100] * preamble_len

        if include_in_labels:
            current_labels += tokens
        else:
            current_labels += [-100] * len(tokens)


    # Save the final chunk
    if len(current_input_ids) > preamble_len and len(current_input_ids) <= MAX_TOKEN_CONTEXT_LENGTH:
        chunks.append({
            "input_ids": current_input_ids,
            "labels": current_labels,
        })

    return chunks


def prep_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if missing

    dataset = load_dataset("ChaiML/50k_duduk_convo")["train"]

    print("Length of dataset:", len(dataset))

    # Process all samples
    processed_data = []
    for sample in tqdm(dataset, desc="Processing dataset"):
        processed_data.extend(process_sample(sample, tokenizer))

    print("Total processed chunks:", len(processed_data))

    # Convert to HuggingFace Dataset
    processed_dataset = Dataset.from_list(processed_data).shuffle(seed=42)

    print("Processed dataset length:", len(processed_dataset))

    split_dataset = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset.save_to_disk("/home/ubuntu/mistral-finetune/mistral_finetune/data/chai_duduk_unmasked/train")
    eval_dataset.save_to_disk("/home/ubuntu/mistral-finetune/mistral_finetune/data/chai_duduk_unmasked/eval")
