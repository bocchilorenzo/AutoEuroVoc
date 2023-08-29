import argparse
import os
import gzip
import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from pagerange import PageRange
from skmultilearn.model_selection import IterativeStratification
import re

def process(args):
    seeds = args.seeds.split(",")
    print(f"Seeds: {seeds}")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for directory in os.listdir(args.data_path):
        if args.langs != "all" and directory not in args.langs.split(","):
            continue
        
        lang = directory

        if not re.match(r"^[A-Za-z]{2}$", lang):
            continue

        print(f"Lang: '{lang}'")
        docSet = set()

        if args.years == "all":
            args.years = [year for year in os.listdir(os.path.join(args.data_path, directory))
                          if os.path.isfile(os.path.join(args.data_path, directory, year))
                          and year.endswith(".json.gz")]
        else:
            args.years = PageRange(args.years).pages
            files_in_directory = [file for file in os.listdir(os.path.join(args.data_path, directory))
                                  if file.endswith(".json.gz")]

            are_any_summarized = ["sum" in file for file in files_in_directory]
            if any(are_any_summarized):
                sum_type = files_in_directory[are_any_summarized.index(True)].split("_", 1)[1]
                args.years = [str(year) + f"_{sum_type}" for year in args.years]
            else:
                args.years = [str(year) + ".json.gz" for year in args.years]

        print(f"Files to process: {', '.join(args.years)}")
        list_inputs = []
        list_labels = []
        for year in args.years:
            currentFile = os.path.join(args.data_path, directory, year)
            print(f"Current file: {currentFile}")
            with gzip.open(currentFile, "rt", encoding="utf-8") as file:
                data = json.load(file)
                for doc in data:
                    labels = data[doc]["eurovoc_classifiers"]

                    list_inputs.append(doc)
                    list_labels.append(labels)

        list_inputs = np.array(list_inputs)
        assert len(list_inputs) == len(list_labels)

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(list_labels)

        finalDict = {}

        for seed in seeds:
            np.random.seed(int(seed))

            finalDict[seed] = {}

            print(f"Splitting files (seed {seed})")

            stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])
            train_idx, aux_idx = next(stratifier.split(list_inputs, y))
            train_ids = list_inputs[train_idx]

            stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.5, 0.5])
            dev_idx, test_idx = next(stratifier.split(list_inputs[aux_idx], y[aux_idx]))
            dev_ids = list_inputs[aux_idx][dev_idx]
            test_ids = list_inputs[aux_idx][test_idx]

            assert(len(set(dev_ids).intersection(set(test_ids))) == 0)
            assert(len(set(dev_ids).intersection(set(train_ids))) == 0)
            assert(len(set(train_ids).intersection(set(test_ids))) == 0)

            finalDict[seed]["train"] = train_ids.tolist()
            finalDict[seed]["test"] = test_ids.tolist()
            finalDict[seed]["dev"] = dev_ids.tolist()

        outputFile = os.path.join(args.output_path, lang + ".json")
        print(f"Writing file {outputFile}")
        with open(outputFile, "w") as fw:
            json.dump(finalDict, fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--langs", type=str, default="en", help="Languages to be processed, separated by a comme (e.g. en,it). Write 'all' to process all the languages.")
    parser.add_argument("--data_path", metavar="FOLDER", type=str, help="Path to the data to process.", required=True)
    parser.add_argument("--output_path", metavar="FOLDER", type=str, help="Folder where to save seeds list (one per language)", required=True)
    parser.add_argument("--years", type=str, default="all", help="Year range to be processed, separated by a minus (e.g. 2010-2020 will get all the years between 2010 and 2020 included) or individual years separated by a comma (to use a single year, simply type it normally like '2016'). Write 'all' to process all the files in the folder.")
    parser.add_argument("--seeds", type=str, default="110", help="Seeds to be used for the randomization and creating the data splits, separated by a comma (e.g. 110,221).")
    args = parser.parse_args()

    process(args)
