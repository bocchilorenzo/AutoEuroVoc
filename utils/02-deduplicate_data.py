import argparse
import json
import gzip
from os import listdir, path, makedirs
from sentence_transformers import SentenceTransformer, util
import numpy as np


def deduplicate_file(args):
    path_initial = args.data_path
    new_path = args.output_path
    makedirs(new_path, exist_ok=True)

    np.random.seed(42)
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    for filename in listdir(path_initial):
        all_docs = {"ids": [], "texts": []}
        if path.isfile(path.join(path_initial, filename)) and filename.endswith(".json.gz"):
            with gzip.open(path.join(path_initial, filename), "rt", encoding="utf-8") as f:
                data = json.load(f)
                for doc in data:
                    all_docs["ids"].append(doc)
                    all_docs["texts"].append(data[doc]["full_text"])
            print(f"Loaded {filename}")
            paraphrases = util.paraphrase_mining(model, all_docs["texts"], show_progress_bar=True, batch_size=args.batch_size)
            deleted = {}
            for paraphrase in paraphrases:
                score, i, j = paraphrase
                if score >= args.threshold:
                    if all_docs["ids"][j] in data:
                        del data[all_docs["ids"][j]]
                        deleted[all_docs["ids"][j]] = all_docs["ids"][i]
            with gzip.open(path.join(new_path, filename), "wt", encoding="utf-8") as f:
                json.dump(data, f)
            if args.save_deleted:
                with open(path.join(new_path, filename[:-8] + "_deleted.json"), "w") as f:
                    json.dump(deleted, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./", help="Path to the folder containing the json.gz files")
    parser.add_argument("--output_path", type=str, default="./deduped", help="Path to the folder where the deduplicated files will be saved")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for the paraphrase mining")
    parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for the paraphrase mining")
    parser.add_argument("--save_deleted", action="store_true", default=False, help="Whether to save the ids of the deleted documents in a JSON file")

    args = parser.parse_args()
    deduplicate_file(args)