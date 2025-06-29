from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
import random
from tqdm import tqdm
import names
import re

from constants import MAX_TOKEN_RESPONSE_LENGTH, MODEL_NAME


random.seed(42)

DATASET_NAME = "flammenai/MahouMix-v1"
USER_NAME = "you"



def unzip_split(data):
    p, c, r = zip(*data)
    return {"prompt": list(p), "chosen": list(c), "rejected": list(r)}


def prep_data():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset(DATASET_NAME, split="train")

    # Reformat examples
    reformatted = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    num_removed = 0
    i = 0
    for example in tqdm(ds, desc="Reformatting examples"):
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        # Note, this dataset has multiple paragraphs of responses, split by newlines. 
        # We only take the first paragraph for simplicity.
        chosen_response = chosen_text.strip().split("\n")[0] + "\n"
        rejected_response = rejected_text.strip().split("\n")[0] + "\n"

        # Filter out long chosen responses
        if len(tokenizer(chosen_response, truncation=True)["input_ids"]) > MAX_TOKEN_RESPONSE_LENGTH:
            num_removed += 1
            continue

        
        post_bio_split = re.split(r'<\|im_end\|>', example["prompt"], maxsplit=1)
        if len(post_bio_split) < 2:
            continue  # No conversation to process
        conversation_text = post_bio_split[1]

        # Step 2: Replace <|im_start|>{name}\n → {name}: 
        conversation_text = re.sub(
            r'<\|im_start\|\>(.*?)\n',
            lambda m: f"{m.group(1).strip()}: ",
            conversation_text
        )

        # Step 3: Replace <|im_end|> (optionally followed by a newline) → \n
        conversation_text = re.sub(r'<\|im_end\|>\n?', '\n', conversation_text)

        reformatted["prompt"].append(conversation_text)
        reformatted["chosen"].append(chosen_response)
        reformatted["rejected"].append(rejected_response)


    print(f"Removed {num_removed} examples with long responses. Kept {len(reformatted['prompt'])} examples.")

    # Zip all examples together
    combined = list(zip(
        reformatted["prompt"],
        reformatted["chosen"],
        reformatted["rejected"]
    ))

    # Shuffle and split
    random.shuffle(combined)
    split_idx = int(0.95 * len(combined))
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    train_split = unzip_split(train_data)
    val_split = unzip_split(val_data)

    out_path_train = "/home/ubuntu/mistral-finetune/mistral_finetune/data/flammenai_mahou/train.json"
    out_path_val = "/home/ubuntu/mistral-finetune/mistral_finetune/data/flammenai_mahou/val.json"

    if not os.path.exists(os.path.dirname(out_path_train)):
        os.makedirs(os.path.dirname(out_path_train))

    with open(out_path_train, "w") as f:
        json.dump(train_split, f, indent=2)
    with open(out_path_val, "w") as f:
        json.dump(val_split, f, indent=2)

    print(f"Saved train data to {out_path_train}.")
    print(f"Saved validation data to {out_path_val}.")
