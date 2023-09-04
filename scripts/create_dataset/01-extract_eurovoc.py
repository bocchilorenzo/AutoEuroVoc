import json
import os
import gzip
from tqdm import tqdm
import argparse

from text_summarizer.lex_utils import init_procedure, add_argument

script_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

def extract_documents(args):
    years = init_procedure(args.data_path, args.output_path, args.years)

    if args.add_mt_do:
        print("Loading MTs and DOs")
        with open(os.path.join(script_folder, "config/domain_labels_position.json"), "r") as fp:
            domain = json.load(fp)
        with open(os.path.join(script_folder, "config/mt_labels_position.json"), "r") as fp:
            microthesaurus = json.load(fp)
        with open(os.path.join(script_folder, "config/mt_labels.json"), "r", encoding="utf-8") as file:
            mt_labels = json.load(file)

    for (year, yearFile) in tqdm(years.items()):
        with gzip.open(os.path.join(args.data_path, yearFile), "rt", encoding="utf-8") as f:
            data = json.load(f)

            to_del = set()
            for doc in data:

                # For each document in the file, only keep those with at least one Eurovoc classifier and without an empty text
                if len(data[doc]["eurovoc_classifiers"]) == 0 or data[doc]["full_text"] == "":
                    to_del.add(doc)
                    continue

                # Add MT and DO labels
                if args.add_mt_do:
                    labels = set(data[doc]["eurovoc_classifiers"])
                    to_add = set()
                    for label in labels:
                        if label in mt_labels:
                            if mt_labels[label] in microthesaurus:
                                to_add.add(mt_labels[label] + "_mt")
                            if mt_labels[label][:2] in domain:
                                to_add.add(mt_labels[label][:2] + "_do")
                    
                    labels = list(labels.union(to_add))
                    data[doc]["eurovoc_classifiers"] = labels

            for doc in to_del:
                del data[doc]

            with gzip.open(os.path.join(args.output_path, year + ".json.gz"), "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract documents containing EuroVoc labels", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_argument(parser, "years")
    add_argument(parser, "data_path")
    add_argument(parser, "output_path")

    parser.add_argument("--add_mt_do", action="store_true", default=False, help="Add the MicroThesaurus and Domain labels.")
    
    args = parser.parse_args()
    extract_documents(args)
