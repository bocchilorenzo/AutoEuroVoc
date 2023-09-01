import argparse
import os
import json
import gzip
from collections import Counter

from text_summarizer.lex_utils import get_years

def process(args):
    if not os.path.exists(args.seed_path):
        print(f"File {seedsFile} does not exist, exiting")
        exit()

    seeds = []
    if args.seeds:
        seeds = args.seeds.split(",")

    seed_data = {}
    with open(args.seed_path, "r") as f:
        seed_data = json.load(f)

    freqs = Counter()
    label_per_doc = {}
    years = get_years(args.years, args.data_path)
    for year in years.items():
        with gzip.open(os.path.join(args.data_path, year[1]), "rt", encoding="utf-8") as file:
            data = json.load(file)

            for doc in data:
                # if doc in label_per_doc:
                #     print(f"WARN: existing doc {doc}")
                label_per_doc[doc] = data[doc]["eurovoc_classifiers"]
                for label in data[doc]["eurovoc_classifiers"]:
                    freqs[label] += 1

    print(freqs.most_common(10))

    for seed in seed_data:
        if seeds and seed not in seeds:
            continue
            
        print(f"### Seed: {seed}")
        freqs_l = {}
        stats = {}
        for l in seed_data[seed]:
            if l not in freqs_l:
                freqs_l[l] = Counter()
            for doc in seed_data[seed][l]:
                # stats[l].update([label for label in label_per_doc[doc]])
                for label in label_per_doc[doc]:
                    freqs_l[l][label] += 1
            print(f"{l}: {len(freqs_l[l].keys())}")

        for l1 in seed_data[seed]:
            for l2 in seed_data[seed]:
                print(f"Intersection ({l1}-{l2})",
                    len(
                        set(freqs_l[l1].keys()).intersection(
                            set(freqs_l[l2].keys())
                        )
                    )
                )

    print("%10s %10s %10s %10s %10s" % ("Class", "Total", "Train", "Dev", "Test"))
    for t in freqs.most_common():
        print("%10s %10d %10d %10d %10d" % (t[0], t[1], freqs_l["train"][t[0]], freqs_l["dev"][t[0]], freqs_l["test"][t[0]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", metavar="FOLDER", type=str, help="Path to the data to process.", required=True)
    parser.add_argument("--years", type=str, default="all", help="Year range to be processed, separated by a minus (e.g. 2010-2020 will get all the years between 2010 and 2020 included) or individual years separated by a comma (to use a single year, simply type it normally like '2016'). Write 'all' to process all the files in the folder.")
    parser.add_argument("--seeds", type=str, default=None, help="Seeds to be used for the randomization and creating the data splits, separated by a comma (e.g. 110,221).")
    parser.add_argument("--seed_path", required=True, type=str, help="JSON file containing seeds information")
    args = parser.parse_args()

    process(args)
