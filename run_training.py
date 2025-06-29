import yaml
import os
import argparse
from train.train_dpo import run_dpo_training


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    print("Loading configuration from:", config_path)
    config = load_config(config_path)
    print(config)

    config["learning_rate"] = float(config["learning_rate"])  # Ensure learning rate is a float
    
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    os.environ["WANDB_ENTITY"] = config["wandb_entity"]    
    os.environ["WANDB_NAME"] = config["wandb_run_name"]

    run_dpo_training(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--config_path",
                        default="configs/dpo_config.yaml",
                        type=str, 
                        help="Path to the configuration file (e.g., configs/dpo_config.yaml)")
    args = parser.parse_args()

    main(args.config_path)