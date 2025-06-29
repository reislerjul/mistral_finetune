import json
import torch
from torch.nn import functional as F
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM
from typing import Dict
import os

from constants import MODEL_NAME


def run_dpo_training(config):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
        device_map="auto",
        offload_folder="offload"
    )

    with open(config["train_dataset_path"]) as f:
        train_data = json.load(f)

    with open(config["eval_dataset_path"]) as f:
        eval_data = json.load(f)

    train_dataset = Dataset.from_dict(train_data)
    eval_dataset = Dataset.from_dict(eval_data)

    dpo_config = DPOConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_steps=50,
        bf16=config["bf16"],
        report_to="wandb",
        run_name=config["wandb_run_name"],
        beta=config["dpo_beta"],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None
    )

    trainer.train()

