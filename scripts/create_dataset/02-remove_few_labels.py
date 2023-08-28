import json
import gzip
from tqdm import tqdm
from os import listdir, path, makedirs
import argparse
from pagerange import PageRange


def remove_few_labels(args):
    path_initial = args.data_path

    makedirs(args.output_path, exist_ok=True)

    if args.years == "all":
        args.years = []
        for filename in listdir(path_initial):
            if (
                path.isfile(path.join(path_initial, filename))
                and filename.endswith(".json.gz")
                and filename.split(".")[0].isdigit()
            ):
                args.years.append(filename)
    else:
        args.years = [
            str(year) + ".json.gz" for year in PageRange(args.years).pages]

    print(f"Working on data from {path_initial}. Files: {', '.join(args.years)}")

    docs_per_label = {}
    tot_docs = 0

    print("Reading data...")
    for year in tqdm(args.years):
        with gzip.open(path.join(path_initial, year), "rt", encoding="utf-8") as f:
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
    print(f"Found {previous_labels_length-len(docs_per_label)}/{previous_labels_length} "
          "labels with less than {args.threshold} documents.")
    print("Removing labels from data...")
    for year in tqdm(args.years):
        with gzip.open(path.join(path_initial, year), "rt", encoding="utf-8") as f:
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
        with gzip.open(path.join(args.output_path, year), "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    print(
        f"Removed {len(deleted_docs)}/{tot_docs} documents ({len(deleted_docs)/tot_docs*100:.2f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove labels associated with few documents from the dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str,
                        default="./data/it/extracted", help="Path to the data folder")
    parser.add_argument("--output_path", type=str,
                        default="./data/it/extracted/few_labels_removed", help="Path to the output folder")
    parser.add_argument("--years", type=str, default="all",
                        help="Years to consider. If not specified, all the years will be considered. Multiple years can be specified either as a comma-separated list (e.g. 2019,2020,2021) or as a range (e.g. 2019-2021).")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Threshold for the number of documents per label")

    args = parser.parse_args()
    remove_few_labels(args)
