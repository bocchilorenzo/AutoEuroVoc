import argparse
import json
import gzip
from os import listdir, path, makedirs
from sentence_transformers import SentenceTransformer, util
import numpy as np

models = {
    "bg": "distiluse-base-multilingual-cased-v2",
    "cs": "distiluse-base-multilingual-cased-v2",
    "da": "distiluse-base-multilingual-cased-v2",
    "de": "distiluse-base-multilingual-cased-v1",
    "el": "distiluse-base-multilingual-cased-v2",
    "en": "all-mpnet-base-v2",
    "es": "distiluse-base-multilingual-cased-v1",
    "et": "distiluse-base-multilingual-cased-v2",
    "fi": "distiluse-base-multilingual-cased-v2",
    "fr": "distiluse-base-multilingual-cased-v1",
    "hu": "distiluse-base-multilingual-cased-v2",
    "it": "distiluse-base-multilingual-cased-v1",
    "lt": "distiluse-base-multilingual-cased-v2",
    "lv": "distiluse-base-multilingual-cased-v2",
    "mt": "distiluse-base-multilingual-cased-v2",
    "nl": "distiluse-base-multilingual-cased-v1",
    "pl": "distiluse-base-multilingual-cased-v1",
    "pt": "distiluse-base-multilingual-cased-v1",
    "ro": "distiluse-base-multilingual-cased-v2",
    "sk": "distiluse-base-multilingual-cased-v2",
    "sl": "distiluse-base-multilingual-cased-v2",
    "sv": "distiluse-base-multilingual-cased-v2"
}

def deduplicate_file(args):
    path_initial = args.data_path
    new_path = args.output_path
    makedirs(new_path, exist_ok=True)

    np.random.seed(42)
    model = SentenceTransformer(models[args.lang])

    print(f"Working on data from {path_initial}. Language: {args.lang}")

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
    parser.add_argument("--lang", type=str, default="it", choices=["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"], help="Language of the documents")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for the paraphrase mining")
    parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for the paraphrase mining")
    parser.add_argument("--save_deleted", action="store_true", default=False, help="Whether to save the ids of the deleted documents in a JSON file")

    args = parser.parse_args()
    deduplicate_file(args)