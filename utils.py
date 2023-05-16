from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, hamming_loss, ndcg_score, precision_score, recall_score, jaccard_score, matthews_corrcoef, multilabel_confusion_matrix
from torch import sigmoid, Tensor, stack, from_numpy
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.ops import focal_loss
from transformers import Trainer
from torch import nn, Tensor, nonzero, sort
import pickle
import json
    
class CustomTrainer(Trainer):
    """
    Custom Trainer to compute the weighted BCE loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_weights = None
        self.use_focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2

    def prepare_labels(self, data_path, language, split, device):
        """
        Set the mlb encoder and the weights for the BCE loss.

        :param data_path: Path to the data.
        :param language: Language of the data.
        :param split: Split of the data.
        :param device: Device to use.
        """
        # Load the weights
        with open(os.path.join(data_path, language, f"split_{split}", "train_labs_count.json"), "r") as weights_fp:
            data = json.load(weights_fp)
            weights = []
            """ # Approach with max weight in case of 0
            for key in data["labels"]:
                # Each weight is the inverse of the frequency of the label. Negative / positive
                weights.append((data["total_samples"] - data["labels"][key])/data["labels"][key] if data["labels"][key] != 0 else None)
            
            # If the weight is None, set it to the maximum weight
            max_weight = max([w for w in weights if w is not None])
            weights = [w if w else max_weight for w in weights] """

            for key in data["labels"]:
                # Each weight is the inverse of the frequency of the label. Negative / positive
                weights.append((data["total_samples"] - data["labels"][key] + 1e-10)/(data["labels"][key] + 1e-10))

            self.custom_weights = Tensor(weights).to(device)
        
    def set_weighted_loss(self):
        """
        Set the loss to the weighted BCE loss.
        """
        self.use_focal_loss = False

    def set_focal_params(self, alpha, gamma):
        """
        Set the focal loss parameters.

        :param alpha: Alpha parameter.
        :param gamma: Gamma parameter.
        """
        self.focal_alpha = alpha
        self.focal_gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function to compute either the focal loss or the weighted BCE loss.

        :param model: Model to use.
        :param inputs: Inputs to the model.
        :param return_outputs: Whether to return the outputs. Default: False.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            logits = outputs.get("logits")
            
            if self.use_focal_loss:
                loss = focal_loss.sigmoid_focal_loss(
                    logits,
                    labels,
                    reduction="mean",
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma
                    )
            else:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.custom_weights)
                loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def sklearn_metrics_core(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False):
    """
    Shared code for the sklearn metrics.

    :param y_true: True labels.
    :param predictions: Predictions.
    :param data_path: Path to the data.
    :param threshold: Threshold to use for the predictions. Default: 0.5.
    :param get_conf_matrix: Whether to get the confusion matrix. Default: False.
    :param get_class_report: Whether to get the classification report. Default: False.
    :return: Initialized variables.
    """
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= threshold).astype(int)

    if get_conf_matrix:
        with open(os.path.join(data_path, 'mlb_encoder.pickle'), 'rb') as f:
            mlb_encoder = pickle.load(f)
        
        # labels = mlb_encoder.inverse_transform(np.ones((1, y_true.shape[1])))[0]
        labels = mlb_encoder.classes_.tolist()
        mlb_conf = multilabel_confusion_matrix(y_true, y_pred)
        conf_matrix = {}
        for i in range(len(labels)):
            conf_matrix[labels[i]] = mlb_conf[i].tolist()
    else:
        conf_matrix = None
    
    if get_class_report:
        class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, digits=4)
        class_report = {
            key: value for key, value in class_report.items() if key.isnumeric()# and value['support'] > 0
        }
    else:
        class_report = None

    to_return = {}

    return probs, y_pred, conf_matrix, class_report, to_return

def sklearn_metrics_full(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False):
    """
    Return all the metrics and the classification report for the predictions.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :param get_conf_matrix: If True, return the confusion matrix. Default: False.
    :param get_class_report: If True, return the classification report. Default: False.
    :return: A dictionary with the metrics and a classification report.
    """
    probs, y_pred, conf_matrix, class_report, to_return = sklearn_metrics_core(y_true, predictions, data_path, threshold, get_conf_matrix, get_class_report)

    averaging=["micro", "macro", "weighted", "samples"]

    to_return["accuracy"] = accuracy_score(y_true, y_pred)
    
    true_labels = [nonzero(Tensor(labels), as_tuple=True)[0] for labels in y_true]
    pred_labels = sort(probs, descending=True)[1][:, :6]
    pk_scores = [np.intersect1d(true, pred).shape[0] / pred.shape[0] + 1e-10 for true, pred in
                    zip(true_labels, pred_labels)]
    rk_scores = [np.intersect1d(true, pred).shape[0] / true.shape[0] + 1e-10 for true, pred in
                    zip(true_labels, pred_labels)]
    f1k_scores = [2 * recall * precision / (recall + precision) for recall, precision in zip(pk_scores, rk_scores)]
    to_return["f1_pyeurovoc"] = sum(f1k_scores) / len(f1k_scores)
    for avg in averaging:
        to_return[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    for avg in averaging:
        to_return[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    
    for avg in averaging:
        to_return[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    
    to_return["hamming_loss"] = hamming_loss(y_true, y_pred)
    
    for avg in averaging:
        to_return[f"jaccard_{avg}"] = jaccard_score(y_true, y_pred, average=avg, zero_division=0)
    
    references = np.array(y_true)
    predictions = np.array(y_pred)
    to_return["matthews_macro"] = np.mean([
        matthews_corrcoef(y_true=references[:, i], y_pred=predictions[:, i], sample_weight=None)
        for i in range(references.shape[1])
    ])
    to_return["matthews_micro"] = matthews_corrcoef(y_true=references.ravel(), y_pred=predictions.ravel())
    
    for avg in averaging:
        try:
            to_return[f"roc_auc_{avg}"] = roc_auc_score(y_true, y_pred, average=avg)
        except ValueError:
            to_return[f"roc_auc_{avg}"] = 0.0
    
    for k in [1, 3, 5, 10]:
        to_return[f"ndcg_{k}"] = ndcg_score(y_true, y_pred, k=k)

    return to_return, class_report, conf_matrix


def sklearn_metrics_single(y_true, predictions, data_path, threshold=0.5, get_conf_matrix=False, get_class_report=False,
    eval_metric=''):
    """
    Return the specified metric and the classification report for the predictions during the training.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :param get_conf_matrix: If True, return the confusion matrix. Default: False.
    :param get_class_report: If True, return the classification report. Default: False.
    :param eval_metric: The metric to use for the evaluation. Default: ''.
    :return: A dictionary with the metric and a classification report.
    """
    _, y_pred, conf_matrix, class_report, to_return = sklearn_metrics_core(y_true, predictions, data_path, threshold, get_conf_matrix, get_class_report)

    if "accuracy" in eval_metric:
        to_return["accuracy"] = accuracy_score(y_true, y_pred)
    elif "f1" in eval_metric:
        to_return[eval_metric] = f1_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "precision" in eval_metric:
        to_return[eval_metric] = precision_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "recall" in eval_metric:
        to_return[eval_metric] = recall_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "hamming" in eval_metric:
        to_return["hamming_loss"] = hamming_loss(y_true, y_pred)
    elif "jaccard" in eval_metric:
        to_return[eval_metric] = jaccard_score(y_true, y_pred, average=eval_metric.split("_")[1], zero_division=0)
    elif "matthews" in eval_metric:
        references = np.array(y_true)
        predictions = np.array(y_pred)
        if eval_metric == "matthews_micro":
            to_return["matthews_micro"] = matthews_corrcoef(y_true=references.ravel(), y_pred=predictions.ravel())
        elif eval_metric == "matthews_macro":
            to_return["matthews_macro"] = np.mean([
                matthews_corrcoef(y_true=references[:, i], y_pred=predictions[:, i], sample_weight=None)
                for i in range(references.shape[1])
            ])
    elif "roc_auc" in eval_metric:
        to_return[eval_metric] = roc_auc_score(y_true, y_pred, average=eval_metric.split("_")[1])
    elif "ndcg" in eval_metric:
        to_return[eval_metric] = ndcg_score(y_true, y_pred, k=eval_metric.split("_")[1])

    return to_return, class_report, conf_matrix

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

def load_data(data_path, lang, data_type):
    """
    Load the data from the specified directory.

    :param data_path: Path to the data.
    :param lang: Language.
    :param data_type: Type of data to load (train or test).
    :return: List of train, dev and test loaders.
    """
    with open("config/seeds.txt", "r") as fp:
        seeds = fp.read().splitlines()

    datasets = []

    for directory in os.listdir(data_path):
        if lang == directory:
            for i in range(len(seeds)):
                if data_type == "train":
                    print("Loading training and dev data from directory {}...".format(os.path.join(data_path, directory, "split_{}".format(i))))

                    # The data is stored in numpy arrays, so it has to be converted to tensors.
                    train_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "train_X.npy")))
                    train_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "train_mask.npy")))
                    train_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "train_y.npy"))).float()

                    assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

                    dev_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "dev_X.npy")))
                    dev_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "dev_mask.npy")))
                    dev_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "dev_y.npy"))).float()

                    assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]

                    dataset_train = TensorDataset(train_X, train_mask, train_y)

                    dataset_dev = TensorDataset(dev_X, dev_mask, dev_y)
                    datasets.append((dataset_train, dataset_dev, train_y.shape[1]))

                elif data_type == "test":
                    print("Loading test data from directory {}...".format(os.path.join(data_path, directory, "split_{}".format(i))))
                    test_X = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "test_X.npy")))
                    test_mask = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "test_mask.npy")))
                    test_y = from_numpy(np.load(os.path.join(data_path, directory, f"split_{i}", "test_y.npy"))).float()

                    assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

                    dataset_test = TensorDataset(test_X, test_mask, test_y)

                    datasets.append((dataset_test))
            break

    return datasets