import yaml
import os
from train.train_dpo import run_dpo_training

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/dpo_config.yaml")
    
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    os.environ["WANDB_ENTITY"] = config["wandb_entity"]    
    os.environ["WANDB_NAME"] = config["wandb_run_name"]

    run_dpo_training(config)

if __name__ == "__main__":
    main()
