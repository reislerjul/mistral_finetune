import re
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from constants import MAX_TOKEN_CONTEXT_LENGTH, MODEL_NAME
from tqdm import tqdm


def preprocess_chat(row, tokenizer, max_length=MAX_TOKEN_CONTEXT_LENGTH):
    full_text = row["input_text"]

    # Split persona and conversation using the first \n
    try:
        persona, conversation = full_text.split("\n", 1)
    except ValueError:
        return None  # skip if format is off

    # Replace "Anonymous user:" with "you:"
    conversation = conversation.replace("Anonymous user:", "you:")

    # Identify character name from the persona section
    match = re.search(r"(.+?)'s Persona", persona)
    if not match:
        return None  # skip if we can't extract character name
    character_name = match.group(1)

    # Tokenize full input (persona + convo)
    full_input = persona + "\n" + conversation

    # padding="max_length"
    inputs = tokenizer(full_input, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    # Tokenize line-by-line to build label mask
    labels = [-100] * len(input_ids)
    current_idx = 0

    for line in conversation.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Prefixes: "you:" or "CharacterName:"
        if line.startswith(f"{character_name}:"):
            line_ids = tokenizer(line, add_special_tokens=False).input_ids
            end_idx = current_idx + len(line_ids)
            if end_idx <= len(labels):
                labels[current_idx:end_idx] = line_ids
        elif line.startswith("you:"):
            pass  # user turn — skip
        else:
            pass  # unexpected line format — also skip

        current_idx += len(tokenizer(line, add_special_tokens=False).input_ids)

    return {
        "input_ids": input_ids,
        "labels": torch.tensor(labels)
    }


def prep_data():
    dataset = load_dataset("ChaiML/20240207_chai_prize_reward_model_data")["train"]
    print("Length of dataset:", len(dataset))

    dataset = dataset.filter(lambda x: x["labels"] == 1)
    print("Filtered dataset length:", len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    processed_examples = []

    id_lengths = []
    label_lengths = []

    for row in tqdm(dataset, desc="Processing dataset"):
        result = preprocess_chat(row, tokenizer)
        if result is not None:
            processed_examples.append(result)
            id_lengths.append(len(result["input_ids"]))
            label_lengths.append(len(result["labels"]))
    
    processed_dataset = Dataset.from_list(processed_examples)

    # print(set(id_lengths))
    # print(set(label_lengths))
    print("Processed dataset length:", len(processed_dataset))

    split_dataset = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset.save_to_disk("/home/ubuntu/mistral-finetune/mistral_finetune/data/chai_reward/train")
    eval_dataset.save_to_disk("/home/ubuntu/mistral-finetune/mistral_finetune/data/chai_reward/eval")
