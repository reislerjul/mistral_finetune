import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import wandb
from constants import MODEL_NAME
import gc



class DataCollatorForCausalLMWithPadding:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features):
        # Separate labels before padding
        labels = [f["labels"] for f in features]
        for f in features:
            del f["labels"]

        batch = self.base_collator(features)

        # Now pad labels
        max_length = max(len(label) for label in labels)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of * self.pad_to_multiple_of
            )

        padded_labels = [
            label + [-100] * (max_length - len(label))
            for label in labels
        ]
        batch["labels"] = torch.tensor(padded_labels)

        return batch



def run_sft_training(config):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForCausalLMWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
        device_map="auto"
    )

    # Load processed datasets
    train_dataset = load_from_disk(config["train_dataset_path"])
    eval_dataset = load_from_disk(config["eval_dataset_path"])

    torch.cuda.empty_cache()
    gc.collect()

    # Define training args
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        bf16=config["bf16"],
        report_to="wandb",
        run_name=config["wandb_run_name"],
        optim="paged_adamw_8bit",
    )

    # Create and launch Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator 
    )

    trainer.train()
