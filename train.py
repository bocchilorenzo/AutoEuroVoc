import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
import yaml
from os import path, makedirs
from utils import sklearn_metrics, data_collator_tensordataset, load_data

language = ""
current_epoch = 0
current_split = 0

def get_metrics(y_true, predictions, threshold=0.5):
    """
    Return the metrics for the predictions.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: Dictionary with the metrics.
    """
    global current_epoch

    metrics, _, _ = sklearn_metrics(
        y_true,
        predictions,
        "",
        threshold,
    )

    current_epoch += 1

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

def start_train():
    """
    Launch the training of the models.
    """
    # Load the configuration for the models of all languages
    with open("config/models.yml", "r") as config_fp:
        config = yaml.safe_load(config_fp)
    
    # Load the seeds for the different splits
    with open("config/seeds.txt", "r") as seeds_fp:
        seeds = seeds_fp.readlines()

    print(f"Working on device: {args.device}")

    # Create the directory for the models
    if not path.exists(args.models_path):
        makedirs(args.models_path)

    # Train the models for all languages
    for lang in config.keys():
        # If a specifiy language is given, skip the others
        if args.lang != "all" and args.lang != lang:
            continue
        
        global language
        language = lang

        # Load the data
        datasets = load_data(args.data_path, lang, "train")

        # Train the models for all splits
        for split_idx, (train_set, dev_set, num_classes) in enumerate(
            datasets
        ):
            # Create the directory for the models of the current language
            makedirs(path.join(args.models_path, lang, str(split_idx)), exist_ok=True)

            # Create the directory for the classification report of the current language
            if args.save_class_report:
                makedirs(path.join(args.models_path, lang, str(split_idx), "classification_report"), exist_ok=True)
            
            global current_split
            current_split = split_idx
            print(f"\nTraining for language: '{lang}' using: '{config[lang]}'...")

            tokenizer = AutoTokenizer.from_pretrained(config[lang])
            
            model = AutoModelForSequenceClassification.from_pretrained(
                config[lang],
                problem_type="multi_label_classification",
                num_labels=num_classes,
                trust_remote_code=True
            )

            # If the device specified via the arguments is "cpu", avoid using CUDA
            # even if it is available
            no_cuda = True if args.device == "cpu" else False

            # Create the training arguments.
            train_args = TrainingArguments(
                path.join(args.models_path, lang, str(split_idx)),
                evaluation_strategy = "epoch",
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                num_train_epochs=args.epochs,
                lr_scheduler_type="linear",
                warmup_steps=len(train_set),
                logging_strategy="epoch",
                logging_dir=path.join(args.models_path, lang, str(split_idx), 'logs'),
                save_strategy = "epoch",
                no_cuda = no_cuda,
                seed = int(seeds[split_idx]),
                load_best_model_at_end=True,
                save_total_limit=1,
                metric_for_best_model="f1_micro",
                optim="adamw_torch",
                optim_args="correct_bias=True",
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                weight_decay=0.01,
            )

            # Create the trainer. It uses a custom data collator to convert the
            # dataset to a compatible dataset.
            trainer = Trainer(
                model,
                train_args,
                train_dataset=train_set,
                eval_dataset=dev_set,
                tokenizer=tokenizer,
                data_collator=data_collator_tensordataset,
                compute_metrics=compute_metrics
            )

            trainer.train()

            # print(f"Best checkpoint path: {trainer.state.best_model_checkpoint}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, default="all", help="Language to train the model on.")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the EuroVoc data.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--models_path", type=str, default="models/", help="Save path of the models")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="Gradient clipping norm.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the prediction confidence.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--save_class_report", default=False, action="store_true", help="Save the classification report.")
    parser.add_argument("--class_report_step", type=int, default=1, help="Number of epochs before creating a new classification report.")
    parser.add_argument("--logging_step", type=int, default=100)

    args = parser.parse_args()

    start_train()
