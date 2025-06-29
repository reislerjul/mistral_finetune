import json
import random
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def combine_datasets():
    random.seed(42)
    dataset_names = ["human_like_dpo", "flammenai_dpo", "flammenai_mahou"]
    combined = {"train": {"prompt": [], "chosen": [], "rejected": []},
                "val": {"prompt": [], "chosen": [], "rejected": []}}

    for split in ["train", "val"]:
        for name in dataset_names:
            path = Path(f"data/{name}/{split}.json")
            print(f"Loading {path}...")
            data = load_json(path)
            for key in ["prompt", "chosen", "rejected"]:
                combined[split][key].extend(data[key])

        # Shuffle all aligned entries together
        combined_zip = list(zip(combined[split]["prompt"],
                                combined[split]["chosen"],
                                combined[split]["rejected"]))
        random.shuffle(combined_zip)
        combined[split]["prompt"], combined[split]["chosen"], combined[split]["rejected"] = zip(*combined_zip)

        # Convert back to lists and save
        combined[split] = {
            "prompt": list(combined[split]["prompt"]),
            "chosen": list(combined[split]["chosen"]),
            "rejected": list(combined[split]["rejected"]),
        }

        output_path = Path(f"data/combined/{split}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(combined[split], output_path)
        print(f"Saved combined {split} to {output_path}")
