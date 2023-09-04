import argparse
from tqdm import tqdm
import os
import gzip
import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from skmultilearn.model_selection import IterativeStratification

from text_summarizer.lex_utils import init_procedure, add_argument

def process(args):
    years = init_procedure(args.data_path, None, args.years)

    seeds = args.seeds.split(",")
    print(f"### Seeds: {', '.join(seeds)}")

    docSet = set()

    list_inputs = []
    list_labels = []

    num_docs = 0

    for (year, yearFile) in (pbar := tqdm(years.items())):
        pbar.set_description(year)
        with gzip.open(os.path.join(args.data_path, yearFile), "rt", encoding="utf-8") as file:
            data = json.load(file)
            for doc in (pbar2 := tqdm(data.keys(), leave=False)):
                pbar2.set_description(doc)
                labels = data[doc]["eurovoc_classifiers"]
                num_docs += 1

                list_inputs.append(doc)
                list_labels.append(labels)

    print(f"### Number of documents: {num_docs}")

    list_inputs = np.array(list_inputs)
    assert len(list_inputs) == len(list_labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(list_labels)

    finalDict = {}

    for seed in seeds:
        np.random.seed(int(seed))

        finalDict[seed] = {}

        print(f"Splitting files (seed {seed})")

        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.5, 0.5])
        train_idx, aux_idx = next(stratifier.split(list_inputs, y))
        train_ids = list_inputs[train_idx]

        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[0.5, 0.5])
        dev_idx, test_idx = next(stratifier.split(list_inputs[aux_idx], y[aux_idx]))
        dev_ids = list_inputs[aux_idx][dev_idx]
        test_ids = list_inputs[aux_idx][test_idx]

        assert(len(set(dev_ids).intersection(set(test_ids))) == 0)
        assert(len(set(dev_ids).intersection(set(train_ids))) == 0)
        assert(len(set(train_ids).intersection(set(test_ids))) == 0)

        finalDict[seed]["train"] = train_ids.tolist()
        finalDict[seed]["dev"] = dev_ids.tolist()
        finalDict[seed]["test"] = test_ids.tolist()
        print(f"Statistics: train={len(finalDict[seed]['train'])}, dev={len(finalDict[seed]['dev'])}, test={len(finalDict[seed]['test'])}")

    print(f"Writing file {args.seed_file}")
    with open(args.seed_file, "w") as fw:
        json.dump(finalDict, fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a list of splits for the data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_argument(parser, "years")
    add_argument(parser, "data_path")
    add_argument(parser, "seed_file")
    add_argument(parser, "seeds")

    args = parser.parse_args()
    process(args)
