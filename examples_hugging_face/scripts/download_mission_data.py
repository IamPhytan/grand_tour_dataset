import argparse
import os
import re
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mission", type=str)
    return parser.parse_args()


# The script is configured to download all the data from one mission
# You can change the mission or use the mission argument and set the dataset_folder to your desired location.
args = parse_arguments()
mission = args.mission or "2024-11-18-13-48-19"

dataset_folder = Path("~/grand_tour_dataset").expanduser()
dataset_folder.mkdir(parents=True, exist_ok=True)

if (dataset_folder / mission).exists():
    print(f"Mission {mission} already downloaded")
    exit()


def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    print(f"Start moving from {cache} to {dataset_folder} !")

    def convert_glob_patterns_to_regex(glob_patterns):
        regex_parts = []
        for pat in glob_patterns:
            # Escape regex special characters except for * and ?
            pat = re.escape(pat)
            # Convert escaped glob wildcards to regex equivalents
            pat = pat.replace(r"\*", ".*").replace(r"\?", ".")
            # Make sure it matches full paths
            regex_parts.append(f".*{pat}$")

        # Join with |
        combined = "|".join(regex_parts)
        return re.compile(combined)

    pattern = convert_glob_patterns_to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]
    tar_files = [f for f in files if f.suffix == ".tar"]

    for source_path in tqdm(tar_files, desc="TAR files"):
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            print(f"Error opening or extracting tar file '{source_path}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {source_path}: {e}")

    other_files = [f for f in files if not f.suffix == ".tar" and f.is_file()]
    for source_path in tqdm(other_files, desc="Copy files"):
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

    print(f"Moved data from {cache} to {dataset_folder} !")


allow_patterns = [f"{mission}/*"]
hugging_face_data_cache_path = snapshot_download(
    repo_id="leggedrobotics/grand_tour_dataset",
    allow_patterns=allow_patterns,
    repo_type="dataset",
    local_dir="/media/damien/data/.cache/grand-tour/",
)
move_dataset(
    hugging_face_data_cache_path, dataset_folder, allow_patterns=allow_patterns
)
