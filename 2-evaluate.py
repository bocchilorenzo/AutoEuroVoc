import argparse
import yaml
from os import path, makedirs, listdir
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, EvalPrediction, TrainingArguments, set_seed
from utils import sklearn_metrics_full, data_collator_tensordataset, load_data
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
    global language

    metrics, class_report, conf_matrix = sklearn_metrics_full(
        y_true,
        predictions,
        path.join(args.data_path, language),
        threshold,
        True,
        True,
        args.parents,
    )

    # Save the classification report
    with open(path.join(
        current_model,
        "evaluation",
        "class_report.json",
        ), "w") as class_report_fp:
        json.dump(class_report, class_report_fp, indent=2)

    # Save the confusion matrix
    with open(path.join(
        current_model,
        "evaluation",
        "conf_matrix.json",
        ), "w") as conf_matrix_fp:
        json.dump(conf_matrix, conf_matrix_fp)
    
    # Save the metrics
    with open(path.join(
        current_model,
        "evaluation",
        "metrics.json"), "w") as metrics_fp:
        json.dump(metrics, metrics_fp, indent=2)
    
    print(metrics)

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
    print(f"Working on device: {args.device}")

    global language
    global current_model
        
    language = args.lang

    # Load the seeds for the different splits
    if args.seeds != "all":
        seeds = args.seeds.split(",")
    else:
        seeds = [name.split("_")[1] for name in listdir(path.join(args.data_path, args.lang)) if "split" in name]

    for seed in seeds:
        if not path.exists(
            path.join(args.models_path, args.lang, seed)
        ):
            print(f"Models for seed {seed} not found. Skipping...")
            continue

        # Load the data
        test_set = load_data(args.data_path, args.lang, "test", seed)

        # Get the last checkpoint
        last_checkpoint = max(
            [
                int(f.split("-")[1])
                for f in listdir(path.join(args.models_path, args.lang, seed))
                if f.startswith("checkpoint-") and path.isdir(path.join(args.models_path, args.lang, seed, f))
            ]
        )
        last_checkpoint = path.join(args.models_path, args.lang, seed, f"checkpoint-{last_checkpoint}")

        # Create the directory for the evaluation output
        makedirs(path.join(last_checkpoint, "evaluation"), exist_ok=True)
        current_model = last_checkpoint

        # Load model and tokenizer
        print(f"\nEvaluating model: '{last_checkpoint}'...")
        set_seed(int(seed))
        tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(last_checkpoint, trust_remote_code=args.trust_remote)
        no_cuda = True if args.device == "cpu" else False

        # Setup the evaluation
        trainer = Trainer(
            args=TrainingArguments(
                path.join(last_checkpoint, "evaluation"),
                per_device_eval_batch_size=args.batch_size,
                seed = int(seed),
                no_cuda = no_cuda,
                report_to="all",
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
    parser.add_argument("--lang", type=str, default="it", help="Language to evaluate the model on.")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the EuroVoc data.")
    parser.add_argument("--seeds", type=str, default="all", help="Seeds to be used to load the data splits, separated by a comma (e.g. 110,221). Use 'all' to use all the data splits.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to evaluate on.")
    parser.add_argument("--parents", type=str, default="none", choices=["none", "add", "builtin"], help="How to handle the parents of the labels. Add them 'artificially' with the 'add' option, or use the 'builtin' option if the labels were added during training.")
    parser.add_argument("--models_path", type=str, default="models/", help="Path of the saved models.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the predictions.")
    parser.add_argument("--trust_remote", action="store_true", default=False, help="Trust the remote code for the model.")
    args = parser.parse_args()

    start_evaluate()
