import argparse
from torch.nn import Sigmoid
from torch import Tensor, stack
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
import yaml
from os import path, makedirs
from load import load_data
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, hamming_loss, ndcg_score

language = ""
current_epoch = 0
current_split = 0

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

def get_metrics(y_true, predictions, threshold=0.5):
    """
    Return the metrics for the predictions.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: Dictionary with the metrics.
    """
    # Convert the predictions to binary
    probs = Sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= threshold).astype(int)

    # Compute the metrics
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    hamming = hamming_loss(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    ndcg_1 = ndcg_score(y_true, y_pred, k=1)
    ndcg_3 = ndcg_score(y_true, y_pred, k=3)
    ndcg_5 = ndcg_score(y_true, y_pred, k=5)
    ndcg_10 = ndcg_score(y_true, y_pred, k=10)

    global current_epoch
    global language
    global current_split

    # Save the classification report
    if args.save_class_report:
        if current_epoch % args.class_report_step == 0:
            class_report = classification_report(
                y_true, y_pred, zero_division=0, output_dict=True, digits=4
            )
            class_report = {
                key: value for key, value in class_report.items() if key.isnumeric() and value['support'] > 0
            }
            with open(path.join(
                            args.save_path,
                            language,
                            "classification_report",
                            f"class_report_train_{language}_{current_split}_{current_epoch}.json",
                        ), "w") as class_report_fp:
                class_report_fp.write(str(class_report))

    current_epoch += 1
    
    # Return as dictionary
    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
        'hamming': hamming,
        'accuracy': accuracy,
        'ndcg_1': ndcg_1,
        'ndcg_3': ndcg_3,
        'ndcg_5': ndcg_5,
        'ndcg_10': ndcg_10
        }
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

    print("Working on device: {}\n".format(args.device))

    # Create the directory for the models
    if not path.exists(args.save_path):
        makedirs(args.save_path)

    # Train the models for all languages
    for lang in config.keys():
        # If a specifiy language is given, skip the others
        if args.lang != "all" and args.lang != lang:
            continue
        
        global language
        language = lang
        # Load the data
        datasets = load_data(args.data_path, lang, "train")

        # Create the directory for the models of the current language
        makedirs(path.join(args.save_path, lang), exist_ok=True)

        # Create the directory for the classification report of the current language
        if args.save_class_report:
            makedirs(path.join(args.save_path, lang, "classification_report"), exist_ok=True)

        # Train the models for all splits
        for split_idx, (train_set, dev_set, num_classes) in enumerate(
            datasets
        ):
            global current_split
            current_split = split_idx
            print(f"\nTraining for language: '{lang}' using: '{config[lang]}'...")

            tokenizer = AutoTokenizer.from_pretrained(config[lang])
            
            model = AutoModelForSequenceClassification.from_pretrained(
                config[lang],
                problem_type="multi_label_classification",
                num_labels=num_classes
            )

            # If the device specified via the arguments is "cpu", avoid using CUDA
            # even if it is available
            no_cuda = True if args.device == "cpu" else False

            # Create the training arguments.
            train_args = TrainingArguments(
                path.join(args.save_path, lang),
                evaluation_strategy = "epoch",
                learning_rate=args.learning_rate,
                max_grad_norm=args.max_grad_norm,
                num_train_epochs=args.epochs,
                lr_scheduler_type="linear",
                warmup_steps=len(train_set),
                logging_strategy="epoch",
                logging_dir=path.join(args.save_path, lang, 'logs'),
                save_strategy = "epoch",
                no_cuda = no_cuda,
                seed = int(seeds[split_idx]),
                load_best_model_at_end=True,
                save_total_limit=3,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, default="all", help="Language to train the model on.")
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the EuroVoc data.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on.")
    parser.add_argument("--save_path", type=str, default="models/", help="Save path of the models")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="Gradient clipping norm.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the prediction confidence.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--save_class_report", default=False, action="store_true", help="Save the classification report.")
    parser.add_argument("--class_report_step", type=int, default=1, help="Number of epochs before creating a new classification report.")
    parser.add_argument("--logging_step", type=int, default=100)

    args = parser.parse_args()

    start_train()
