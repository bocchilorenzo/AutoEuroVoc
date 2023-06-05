import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, EvalPrediction, AutoTokenizer, set_seed, Trainer
import yaml
from os import path, makedirs
from utils import sklearn_metrics_single, sklearn_metrics_full, data_collator_tensordataset, load_data, CustomTrainer
import json

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
    global language
    global current_split

    metrics, class_report, _ = sklearn_metrics_full(
        y_true,
        predictions,
        "",
        threshold,
        False,
        args.save_class_report,
    ) if args.full_metrics else sklearn_metrics_single(
        y_true,
        predictions,
        "",
        threshold,
        False,
        args.save_class_report,
        eval_metric=args.eval_metric,
    )

    if args.save_class_report:
        if current_epoch % args.class_report_step == 0:
            with open(path.join(
                args.models_path,
                language,
                str(current_split),
                "train_reports",
                f"class_report_{current_epoch}.json",
            ), "w") as class_report_fp:
                class_report.update(metrics)
                json.dump(class_report, class_report_fp, indent=2)

    current_epoch += 1

    return metrics


def compute_metrics(p: EvalPrediction):
    """
    Compute the metrics for the predictions during the training.

    :param p: EvalPrediction object.
    :return: Dictionary with the metrics.
    """
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
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

    print(f"\nArguments: {vars(args)}")

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

        print(f"\nTraining for language: '{lang}' using: '{config[lang]}'...")

        # Train the models for all splits
        for split_idx in range(len(seeds)):
            global current_split
            current_split = split_idx

            # Load the data
            train_set, dev_set, num_classes = load_data(args.data_path, lang, "train", split_idx)

            # Create the directory for the models of the current language
            makedirs(path.join(args.models_path, lang,
                     str(split_idx)), exist_ok=True)

            # Create the directory for the classification report of the current language
            if args.save_class_report:
                makedirs(path.join(args.models_path, lang, str(
                    split_idx), "train_reports"), exist_ok=True)

            set_seed(int(seeds[split_idx]))

            tokenizer = AutoTokenizer.from_pretrained(config[lang])

            with open(path.join(args.data_path, language, f"split_{split_idx}", "train_labs_count.json"), "r") as weights_fp:
                data = json.load(weights_fp)
                labels = list(data["labels"].keys())

            model = AutoModelForSequenceClassification.from_pretrained(
                config[lang],
                problem_type="multi_label_classification",
                num_labels=num_classes,
                id2label={id_label:label for id_label, label in enumerate(labels)},
                label2id={label:id_label for id_label, label in enumerate(labels)},
                trust_remote_code=args.trust_remote,
            )

            # If the device specified via the arguments is "cpu", avoid using CUDA
            # even if it is available
            no_cuda = True if args.device == "cpu" else False

            # Create the training arguments.
            train_args = TrainingArguments(
                path.join(args.models_path, lang, str(split_idx)),
                evaluation_strategy="epoch",
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                num_train_epochs=args.epochs,
                lr_scheduler_type="linear",
                warmup_steps=len(train_set),
                logging_strategy="epoch",
                logging_dir=path.join(
                    args.models_path, lang, str(split_idx), 'logs'),
                save_strategy="epoch",
                no_cuda=no_cuda,
                seed=int(seeds[split_idx]),
                load_best_model_at_end=True,
                save_total_limit=1,
                metric_for_best_model=args.eval_metric,
                optim="adamw_torch",
                optim_args="correct_bias=True",
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                weight_decay=0.01,
                report_to="all",
            )

            # Create the trainer. It uses a custom data collator to convert the
            # dataset to a compatible dataset.

            if args.custom_loss:
                trainer = CustomTrainer(
                    model,
                    train_args,
                    train_dataset=train_set,
                    eval_dataset=dev_set,
                    tokenizer=tokenizer,
                    data_collator=data_collator_tensordataset,
                    compute_metrics=compute_metrics
                )

                trainer.prepare_labels(
                    args.data_path, lang, split_idx, args.device)

                if args.weighted_loss:
                    trainer.set_weighted_loss()
            else:
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
    # fmt: off
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, default="all", help="Language to train the model on.")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the EuroVoc data.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to train on.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="Gradient clipping norm.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the prediction confidence.")
    parser.add_argument("--custom_loss", action="store_true", default=False, help="Enable the custom loss (focal loss by default).")
    parser.add_argument("--weighted_loss", action="store_true", default=False, help="Enable the weighted bcewithlogits loss. Only works if the custom loss is enabled.")
    parser.add_argument("--eval_metric", type=str, default="f1_micro", choices=[
        'loss', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples',
        'matthews_macro', 'matthews_micro',
        'roc_auc_micro', 'roc_auc_macro', 'roc_auc_weighted', 'roc_auc_samples',
        'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples',
        'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
        'hamming_loss', 'accuracy', 'ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10'],
        help="Evaluation metric to use on the validation set.")
    parser.add_argument("--full_metrics", action="store_true", default=False, help="Compute all the metrics during the evaluation.")
    parser.add_argument("--trust_remote", action="store_true", default=False, help="Trust the remote code for the model.")
    parser.add_argument("--models_path", type=str, default="models/", help="Save path of the models.")
    parser.add_argument("--save_class_report", action="store_true", default=False, help="Save the classification report.")
    parser.add_argument("--class_report_step", type=int, default=1, help="Number of epochs before creating a new classification report.")

    args = parser.parse_args()

    start_train()
