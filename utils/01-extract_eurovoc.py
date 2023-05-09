import json
from os import listdir, path
import gzip
import argparse

def extract_documents(args):
    path_docs = args.data_path
    years = [name for name in listdir(path_docs) if path.isfile(path.join(path_docs, name)) and name.endswith(".json.gz")]
    final_path = args.output_path

    for year in years:
        with gzip.open(path.join(path_docs, year), "rt", encoding="utf-8") as f:
            data = json.load(f)
            to_del = set()
            for doc in data:
                if len(data[doc]["eurovoc_classifiers"]) == 0 or data[doc]["full_text"] == "":
                    to_del.add(doc)
            for doc in to_del:
                del data[doc]
            with gzip.open(path.join(final_path, year), "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./", help="Path to the folder containing the .json.gz files")
    parser.add_argument("--output_path", type=str, default="./", help="Path to the folder where the output files will be saved")
    
    args = parser.parse_args()
    extract_documents(args)