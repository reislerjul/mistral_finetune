import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import upload_file


def push_checkpoints(base_path, hf_user):
    folder_name = os.path.basename(os.path.normpath(base_path))
    readme_path = os.path.join(base_path, "README.md")

    # Find all checkpoints
    checkpoints = sorted(
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("checkpoint-")
    )

    for ckpt in checkpoints:
        ckpt_path = os.path.join(base_path, ckpt)
        step = ckpt.split("-")[-1]
        repo_id = f"{hf_user}/{folder_name}-step-{step}"

        print(f"\n📤 Pushing checkpoint {ckpt_path} to {repo_id}...")

        model = AutoModelForCausalLM.from_pretrained(ckpt_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

        # Optionally upload README
        if os.path.exists(readme_path):
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model"
            )
            print("📝 Uploaded README.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push checkpoints to Hugging Face Hub.")
    parser.add_argument("--base_path",
                        default="/home/ubuntu/mistral-finetune/mistral_finetune/models/mistral-dpo/",
                        type=str, 
                        help="Path to folder containing checkpoints")
    parser.add_argument("--hf_user", type=str, default="jreisler", help="Your Hugging Face username or org (e.g., jreisler)")

    args = parser.parse_args()
    push_checkpoints(args.base_path, args.hf_user)
