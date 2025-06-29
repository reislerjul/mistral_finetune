import yaml
import os
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

        

def main(config_path, training_strategy):
    print("Loading configuration from:", config_path)
    config = load_config(config_path)
    print(config)

    config["learning_rate"] = float(config["learning_rate"])  # Ensure learning rate is a float
    
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    os.environ["WANDB_ENTITY"] = config["wandb_entity"]    
    os.environ["WANDB_NAME"] = config["wandb_run_name"]

    if training_strategy == "sft":
        from train.train_sft import run_sft_training
        print("Running SFT training...")
        run_sft_training(config)
    elif training_strategy == "dpo":
        from train.train_dpo import run_dpo_training
        print("Running DPO training...")
        run_dpo_training(config)
    else:
        raise ValueError(f"Unknown training strategy: {training_strategy}. Choose 'sft' or 'dpo'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--config_path",
                        default="configs/dpo_config.yaml",
                        type=str, 
                        help="Path to the configuration file (e.g., configs/dpo_config.yaml)")
    parser.add_argument("--training_strategy",
                        default="dpo",
                        type=str, 
                        help="Training strategy to use (e.g., dpo, sft)")                        
    args = parser.parse_args()

    main(args.config_path, args.training_strategy)