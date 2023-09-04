import json
import os
import gzip
from tqdm import tqdm
import argparse

from text_summarizer.lex_utils import init_procedure, add_argument

def remove_few_labels(args):
    years = init_procedure(args.data_path, args.output_path, args.years)

    docs_per_label = {}
    tot_docs = 0

    print("Reading data...")
    for (year, yearFile) in tqdm(years.items()):
        with gzip.open(os.path.join(args.data_path, yearFile), "rt", encoding="utf-8") as f:
            data = json.load(f)
            for doc in data:
                for label in data[doc]["eurovoc_classifiers"]:
                    if label not in docs_per_label:
                        docs_per_label[label] = 0
                    docs_per_label[label] += 1
                tot_docs += 1

    previous_labels_length = len(docs_per_label)
    for label in list(docs_per_label.keys()):
        if docs_per_label[label] < args.threshold:
            del docs_per_label[label]

    deleted_docs = set()
    print(f"Found {previous_labels_length - len(docs_per_label)}/{previous_labels_length} "
          f"labels with less than {args.threshold} documents.")

    print("Removing labels from data...")
    for (year, yearFile) in tqdm(years.items()):
        with gzip.open(os.path.join(args.data_path, yearFile), "rt", encoding="utf-8") as f:
            data = json.load(f)
            for doc in list(data.keys()):
                for label in list(data[doc]["eurovoc_classifiers"]):
                    if label not in docs_per_label:
                        label_index = data[doc]["eurovoc_classifiers"].index(
                            label)
                        del data[doc]["eurovoc_classifiers"][label_index]
                if len(data[doc]["eurovoc_classifiers"]) == 0:
                    deleted_docs.add((doc, year))
                    del data[doc]
        with gzip.open(os.path.join(args.output_path, year + ".json.gz"), "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    print(f"Removed {len(deleted_docs)}/{tot_docs} documents ({len(deleted_docs) / tot_docs * 100 : .2f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove labels associated with few documents from the dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_argument(parser, "years")
    add_argument(parser, "data_path")
    add_argument(parser, "output_path")

    parser.add_argument("--threshold", metavar="NUM", type=int, default=10, help="Threshold for the number of documents per label")

    args = parser.parse_args()
    remove_few_labels(args)
