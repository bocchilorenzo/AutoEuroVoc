import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, EvalPrediction, AutoTokenizer, set_seed, Trainer
import yaml
from os import path, makedirs
from utils import sklearn_metrics_single, sklearn_metrics_full, data_collator_tensordataset, load_data, CustomTrainer
import json
from optuna import Trial, samplers, create_study, visualization

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


def objective(trial: Trial, tune_params, model, tokenizer, train_set, dev_set):
    """
    Objective function for the hyperparameter tuning.

    :param trial: Trial object.
    :param tune_params: Dictionary with the parameters for the tuning.
    :param model: Model to be tuned.
    :param tokenizer: Tokenizer for the model.
    :param train_set: Training set.
    :param dev_set: Evaluation set during the training.
    :return: Metric for the current trial.
    """
    global language
    global current_split

    training_args = TrainingArguments(
        output_dir=tune_params["output_dir"],
        learning_rate=trial.suggest_float(
            "learning_rate", low=1e-5, high=6e-5, step=1e-5),
        max_grad_norm=tune_params["max_grad_norm"],
        weight_decay=trial.suggest_categorical(
            "weight_decay", choices=[0.0001, 0.001, 0.01, 0.1, 1.0]),
        num_train_epochs=trial.suggest_int(
            "num_train_epochs", low=30, high=100),
        lr_scheduler_type="linear",
        warmup_steps=len(train_set),
        seed=tune_params["seed"],
        save_total_limit=1,
        no_cuda=tune_params["cuda_choice"],
        metric_for_best_model=tune_params["eval_metric"],
        optim="adamw_torch",
        optim_args=trial.suggest_categorical(
            "optim_args", choices=["correct_bias=True", "correct_bias=False"]),
        per_device_train_batch_size=tune_params["batch_size"],
        per_device_eval_batch_size=tune_params["batch_size"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="all",
    )
    if tune_params["custom_loss"]:
        trainer = CustomTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=dev_set,
            data_collator=data_collator_tensordataset,
            compute_metrics=compute_metrics
        )

        trainer.prepare_labels(args.data_path, language,
                               current_split, args.device)

        if tune_params["weighted_loss"]:
            trainer.set_weighted_loss()
        else:
            trainer.set_focal_params(
                alpha=trial.suggest_float("alpha", low=0.15, high=0.95, step=0.1),
                gamma=trial.suggest_float("gamma", low=1, high=9, step=1)
            )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=dev_set,
            data_collator=data_collator_tensordataset,
            compute_metrics=compute_metrics
        )
    result = trainer.train()
    # This return does not work
    return result[tune_params["eval_metric"]]


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
            global current_split
            current_split = split_idx

            # Create the directory for the models of the current language
            makedirs(path.join(args.models_path, lang,
                     str(split_idx)), exist_ok=True)

            # Create the directory for the classification report of the current language
            if args.save_class_report:
                makedirs(path.join(args.models_path, lang, str(
                    split_idx), "train_reports"), exist_ok=True)

            print(
                f"\nTraining for language: '{lang}' using: '{config[lang]}'...")

            print(f"\nArguments: {vars(args)}")

            set_seed(int(seeds[split_idx]))

            tokenizer = AutoTokenizer.from_pretrained(config[lang])

            model = AutoModelForSequenceClassification.from_pretrained(
                config[lang],
                problem_type="multi_label_classification",
                num_labels=num_classes,
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

            if args.autotune:
                # Set the parameters for the automated tuning
                tune_params = {
                    "seed": int(seeds[split_idx]),
                    "eval_metric": args.eval_metric,
                    "output_dir": path.join(args.models_path, lang, str(split_idx), "tuning"),
                    "cuda_choice": no_cuda,
                    "max_grad_norm": args.max_grad_norm,
                    "batch_size": args.batch_size,
                    "custom_loss": args.custom_loss,
                    "weighted_loss": args.weighted_loss,
                }

                # Create the directory for the models of the current language
                makedirs(tune_params["output_dir"], exist_ok=True)

                # We create the study with the direction of the optimization
                study = create_study(
                    study_name='hyper-parameter-search',
                    direction=args.tune_direction,
                    sampler=samplers.TPESampler(seed=tune_params["seed"]),
                    storage=f"sqlite:///{path.join(tune_params['output_dir'], 'optimization.db')}",
                )

                # Optimize the objective using 15 different trials
                study.optimize(lambda trial: objective(
                    trial, tune_params, model, tokenizer, train_set, dev_set), n_trials=30)

                with open(path.join(tune_params["output_dir"], "optimization_results.txt"), "w") as optim_fp:
                    optim_fp.write(
                        f"Best {tune_params['eval_metric']} value: {str(study.best_value)}\nBest hyperparameters: {str(study.best_params)}\nBest trial: {str(study.best_trial)}")

                # Gives the best evaluation parameter value
                print(
                    f"Best {tune_params['eval_metric']} value: {str(study.best_value)}")

                # Gives the best hyperparameter values to get the best evaluation parameter value
                print(f"Best hyperparameters: {str(study.best_params)}")

                # Return info about best Trial such as start and end datetime, hyperparameters
                print(f"Best trial: {str(study.best_trial)}")

                fig = visualization.plot_intermediate_values(study)
                fig.write_image(
                    path.join(tune_params["output_dir"], "intermediate_values.png"))

                fig = visualization.plot_optimization_history(study)
                fig.write_image(
                    path.join(tune_params["output_dir"], "optimization_history.png"))

                study.trials_dataframe().to_csv(
                    path.join(tune_params["output_dir"], "trials_dataframe.csv"))

                # Use the best hyperparameters to train the model
                for k, v in study.best_trial.hyperparameters.items():
                    if k != "alpha" and k != "gamma":
                        setattr(train_args, k, v)
            else:
                study = None
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
                    if study:
                        trainer.set_focal_params(
                            alpha=study.best_trial.hyperparameters["alpha"],
                            gamma=study.best_trial.hyperparameters["gamma"]
                        )
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
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of the dataset.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--max_grad_norm", type=int, default=5, help="Gradient clipping norm.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the prediction confidence.")
    parser.add_argument("--custom_loss", action="store_true", default=False, help="Enable the custom loss (focal loss by default).")
    parser.add_argument("--weighted_loss", action="store_true", default=False, help="Enable the weighted bcewithlogits loss. Only works if the custom loss is enabled.")
    parser.add_argument("--autotune", action="store_true", default=False, help="Automatically look for the best hyperparameters.")
    parser.add_argument("--tune_direction", type=str, default="maximize", choices=["maximize", "minimize"], help="Direction of the optimization.")
    parser.add_argument("--eval_metric", type=str, default="f1_micro", choices=[
        'loss', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples',
        'matthews_macro', 'matthews_micro',
        'roc_auc_micro', 'roc_auc_macro', 'roc_auc_weighted', 'roc_auc_samples',
        'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples',
        'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples',
        'hamming_loss', 'accuracy', 'ndcg_1', 'ndcg_3', 'ndcg_5', 'ndcg_10'],
        help="Evaluation metric to use for the optimization.")
    parser.add_argument("--full_metrics", action="store_true", default=False, help="Compute all the metrics during the evaluation.")
    parser.add_argument("--trust_remote", action="store_true", default=False, help="Trust the remote code for the model.")
    parser.add_argument("--models_path", type=str, default="models/", help="Save path of the models")
    parser.add_argument("--save_class_report", action="store_true", default=False, help="Save the classification report.")
    parser.add_argument("--class_report_step", type=int, default=1, help="Number of epochs before creating a new classification report.")

    args = parser.parse_args()

    start_train()
