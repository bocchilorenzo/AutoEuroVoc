import argparse
import yaml
from os import path, makedirs, listdir
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, EvalPrediction, TrainingArguments
from utils import sklearn_metrics, data_collator_tensordataset, load_data
import json

language = ""
current_model = ""

def get_metrics(y_true, predictions, threshold=0.5):
    """
    Return the metrics for the predictions.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: Dictionary with the metrics.
    """
    global current_model

    metrics, class_report = sklearn_metrics(
        y_true,
        predictions,
        threshold,
    )

    # Save the classification report
    with open(path.join(
        current_model,
        "evaluation",
        "class_report.json",
        ), "w") as class_report_fp:
        json.dump(class_report, class_report_fp, indent=2)
    
    # Save the metrics
    with open(path.join(
        current_model,
        "evaluation",
        "metrics.json"), "w") as metrics_fp:
        json.dump(metrics, metrics_fp, indent=2)

    return metrics

def compute_metrics(p: EvalPrediction):
    """
    Compute the metrics for the predictions during the training.

    :param p: EvalPrediction object.
    :return: Dictionary with the metrics.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = get_metrics(p.label_ids, preds, args.threshold)
    return result

def start_evaluate():
    """
    Launch the evaluation of the models.
    """
    # Load the configuration for the models of all languages
    with open("config/models.yml", "r") as config_fp:
        config = yaml.safe_load(config_fp)

    print(f"Working on device: {args.device}")

    global language
    global current_model

    # Evaluate the models for all languages
    for lang in config.keys():
        # If a specifiy language is given, skip the others
        if args.lang != "all" and args.lang != lang:
            continue
        
        language = lang

        # Load the data
        datasets = load_data(args.data_path, lang, "test")

        for split_idx, (test_set) in enumerate(datasets):
            if not path.exists(
                path.join(args.models_path, lang, str(split_idx))
            ):
                break

            # Get the last checkpoint
            last_checkpoint = max(
                [
                    int(f.split("-")[1])
                    for f in listdir(path.join(args.models_path, lang, str(split_idx)))
                    if f.startswith("checkpoint-") and path.isdir(path.join(args.models_path, lang, str(split_idx), f))
                ]
            )
            last_checkpoint = path.join(args.models_path, lang, str(split_idx), f"checkpoint-{last_checkpoint}")

            # Create the directory for the evaluation output
            makedirs(path.join(last_checkpoint, "evaluation"), exist_ok=True)
            current_model = last_checkpoint

            # Load model and tokenizer
            print(f"\nEvaluating model: '{last_checkpoint}'...")
            tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(last_checkpoint)
            no_cuda = True if args.device == "cpu" else False

            # Setup the evaluation
            trainer = Trainer(
                args=TrainingArguments(
                    path.join(last_checkpoint, "evaluation"),
                    per_device_eval_batch_size=args.batch_size,
                    no_cuda = no_cuda
                ),
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator_tensordataset,
                compute_metrics=compute_metrics
            )

            # Evaluate the model
            model.eval()
            trainer.predict(test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, default="all", help="Language to train the model on.")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the EuroVoc data.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--models_path", type=str, default="models/", help="Path of the saved models.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the predictions.")
    args = parser.parse_args()

    start_evaluate()
