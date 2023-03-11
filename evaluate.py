import argparse
import evaluate
import yaml
from torch import Tensor, stack
from os import path, makedirs
from load import load_data
from transformers import AutoModelForSequenceClassification

def data_collator_tensordataset(features):
    """
    Custom data collator for datasets of the type TensorDataset.

    :param features: List of features.
    :return: Batch.
    """
    batch = {}
    batch['input_ids'] = stack([f[0] for f in features])
    batch['attention_mask'] = stack([f[1] for f in features])
    batch['labels'] = stack([f[2] for f in features])
    
    return batch

def start_evaluate():
    """
    Launch the evaluation of the models.
    """
    # Load the configuration for the models of all languages
    with open("../config/models.yml", "r") as config_fp:
        config = yaml.safe_load(config_fp)

    print("Working on device: {}\n".format(args.device))

    # Create the directory for the output
    if not path.exists(args.output_path):
        makedirs(args.output_path)

    # Evaluate the models for all languages
    for lang in config.keys():
        # If a specifiy language is given, skip the others
        if args.lang != "all" and args.lang != lang:
            continue
        
        # Load the data
        datasets = load_data(args.data_path, lang, "test")

        for split_idx, (test_loader) in enumerate(datasets[:2]):
            if not path.exists(
                path.join(args.models_path, lang, "model_{}.pt".format(split_idx))
            ):
                break

            print("\nEvaluating model: '{}'...".format("model_{}.pt".format(split_idx)))

            print("Total steps: {}".format(len(test_loader)))

            # TRY A TRAINING FIRST AND THEN ADD THE EVALUATION


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="all", help="Language to train the model on.")
    parser.add_argument("--data_path", type=str, default="../data/", help="Path to the EuroVoc data.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--models_path", type=str, default="models", help="Path of the saved models.")
    parser.add_argument("--output_path", type=str, default="output", help="Models evaluation output path.")
    parser.add_argument("--save_class_report", default=False, action="store_true", help="Save the classification report.")
    parser.add_argument("--logging_step", type=int, default=100)
    args = parser.parse_args()

    start_evaluate()
