import json
from os import path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels' JSON file")
    args = parser.parse_args()

    with open(path.join(args.model_dir, "config.json"), "r") as f:
        config = json.load(f)

    with open(args.labels, "r") as f:
        train_labs_count = json.load(f)
        labels = list(train_labs_count["labels"].keys())


    config["id2label"]={id_label:label for id_label, label in enumerate(labels)}
    config["label2id"]={label:id_label for id_label, label in enumerate(labels)}

    with open(path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)