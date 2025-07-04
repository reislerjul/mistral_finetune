import yaml
import os
import argparse


def main(dataset_name):
    if dataset_name == "human_like_dpo":
        print("Preparing Human-Like DPO dataset...")
        from data_prep.human_like_dpo import prep_data
        prep_data()
    elif dataset_name == "flammenai_dpo":
        print("Preparing FlammenAI DPO dataset...")
        from data_prep.flammenai_dpo import prep_data
        prep_data()
    elif dataset_name == "flammenai_mahou":
        print("Preparing FlammenAI Mahou dataset...")
        from data_prep.flammenai_mahou import prep_data
        prep_data()
    elif dataset_name == "combined":
        print("Combining all datasets...")
        from data_prep.combine_datasets import combine_datasets
        combine_datasets()
    elif dataset_name == "chai_reward":
        print("Preparing Chai Reward dataset...")
        from data_prep.chai_reward import prep_data
        prep_data()
    elif dataset_name == "chai_duduk":
        print("Preparing Chai Duduk dataset...")
        from data_prep.chai_duduk import prep_data
        prep_data()
    elif dataset_name == "chai_duduk_unmasked":
        print("Preparing Chai Duduk Unmasked dataset...")
        from data_prep.chai_duduk_unmasked import prep_data
        prep_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'human_like_dpo' or 'flammenai_dpo'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument("--dataset_name",
                        default="human_like_dpo",
                        type=str, 
                        help="Name of dataset to process (e.g., human_like_dpo, flammenai_dpo)")
    args = parser.parse_args()

    main(args.dataset_name)
