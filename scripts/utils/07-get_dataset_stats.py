import json
import gzip
from os import path, listdir, makedirs
from tqdm import tqdm
import argparse

def get_stats(args):
    langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
    if not path.isdir(args.full) or not path.isdir(args.truncated):
        raise ValueError('Invalid paths')
    
    stats = {}
    
    for lang in langs:
        full_files = sorted([file for file in listdir(path.join(args.full, lang)) if file.endswith('.json.gz')])
        truncated_files = sorted([file for file in listdir(path.join(args.truncated, lang)) if file.endswith('.json.gz')])

        print(f"Processing {lang}...")

        labels_truncated = set()
        full_labels = set()
        length_truncated = 0
        length_full = 0

        for full_file, truncated_file in tqdm(zip(full_files, truncated_files), total=len(full_files)):
            with gzip.open(path.join(args.truncated, lang, truncated_file), 'r') as f:
                truncated_data = json.load(f)
                length_truncated += len(truncated_data)
                for doc in truncated_data:
                    for label in truncated_data[doc]["eurovoc_classifiers"]:
                        labels_truncated.add(label)
                

            with gzip.open(path.join(args.full, lang, full_file), 'r') as f:
                full_data = json.load(f)
                length_full += len(full_data)
                for doc in full_data:
                    for label in full_data[doc]["eurovoc_classifiers"]:
                        full_labels.add(label)

        stats[lang] = {
            "full_length": length_full,
            "final_length": length_truncated,
            "full_labels": len(full_labels),
            "final_labels": len(labels_truncated),
            "removed_labels": len(full_labels - labels_truncated),
            "removed_docs": length_full - length_truncated
        }

        print(f"Stats for {lang}:")
        print(f"Full length: {length_full}")
        print(f"Final length: {length_truncated}")
        print(f"Full labels: {len(full_labels)}")
        print(f"Final labels: {len(labels_truncated)}")
        print(f"Removed labels: {len(full_labels - labels_truncated)}")
        print(f"Removed docs: {length_full - length_truncated}")

    makedirs(args.output, exist_ok=True)
    with open(path.join(args.output, "stats.json"), 'w', encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=str, required=True, help='Path to the folder for the full data, organized by language')
    parser.add_argument('--truncated', type=str, required=True, help='Path with the truncated data, organized by language')
    parser.add_argument('--output', type=str, required=True, help='Path for the output folder')
    args = parser.parse_args()
    
    get_stats(args)