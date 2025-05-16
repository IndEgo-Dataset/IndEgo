# Download the category of interest from HF

import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "vivek9chavan/IndEgo_Demo"
CATEGORY_OPTIONS = [
    "1_Assembly",
    "1_Disassembly",
    "2_Inspection_Repair",
    "3_Logistics_Organization",
    "4_Woodworking",
    "5_Miscellaneous",
    "6_Tools_Objects_in_Context",
    "7_Tools_Objects_demo",
    "8_Singular_Actions",
    "Mistake_Detection"
]

def download_category(category_name, output_dir="downloads"):
    if category_name not in CATEGORY_OPTIONS:
        raise ValueError(f"Category '{category_name}' is not valid. Choose from:\n{CATEGORY_OPTIONS}")
    
    print(f"Downloading category: {category_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    category_files = [f for f in all_files if f.startswith(category_name + "/")]

    for fpath in category_files:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fpath,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {fpath} → {local_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download full category folders from IndEgo on Hugging Face")
    parser.add_argument("--category", type=str, required=True, help="Name of the category folder to download")
    parser.add_argument("--output_dir", type=str, default="downloads", help="Where to save the files")
    args = parser.parse_args()

    download_category(args.category, args.output_dir)
