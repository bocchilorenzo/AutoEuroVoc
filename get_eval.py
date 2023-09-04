import argparse
import os
import json

from scripts.create_dataset.text_summarizer.lex_utils import add_argument

# tc, mt, do

def process(args):

    seeds = []
    if args.seeds:
        seeds = args.seeds.split(",")

    globalValues = {}
    for seed in seeds:
        seedPath = os.path.join(args.data_path, seed)
        if not os.path.exists(seedPath):
            print(f"Path {seedPath} does not exist")
            continue

        for file in os.listdir(seedPath):
            if not os.path.isdir(os.path.join(seedPath, file)):
                continue

            evaluationFile = os.path.join(seedPath, file, "evaluation", "metrics.json")
            if not os.path.exists(evaluationFile):
                continue

            with open(evaluationFile, "r") as f:
                results = json.load(f)

            for key in results:
                if key not in globalValues:
                    globalValues[key] = []
                globalValues[key].append(results[key])

    avgValues = {}
    for k in globalValues:
        avgValues[k] = sum(globalValues[k]) / len(globalValues[k])

    for k in avgValues:
        print(f"{k}: {avgValues[k]:.3f}")
    # print(avgValues)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the average of seeds.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_argument(parser, "data_path")
    add_argument(parser, "seeds")

    args = parser.parse_args()
    process(args)
