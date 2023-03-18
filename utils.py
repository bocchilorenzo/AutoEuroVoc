from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, hamming_loss, ndcg_score, precision_score, recall_score
from torch import sigmoid, Tensor, stack, from_numpy
import os 
import numpy as np
from torch.utils.data import TensorDataset
import pickle
import json

def sklearn_metrics(y_true, predictions, data_path, language, threshold=0.5):
    """
    Return the metrics and classification report for the predictions.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param data_path: Path to the data.
    :param language: Language.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: A dictionary with the metrics and a classification report.
    """
    # Load the label encoder
    with open(os.path.join(data_path, language, "mlb_encoder.pickle"), "rb") as file:
        mlb_encoder = pickle.load(file)
    
    # The domains and microthesaurus labels are loaded from the json files
    with open("config/domain_labels_position.json", "r") as fp:
        domain = json.load(fp)
    with open("config/mt_labels_position.json", "r") as fp:
        microthesaurus = json.load(fp)
    with open("config/mt_labels.json", "r", encoding="utf-8") as file:
        mt_labels = json.load(file)
    
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= threshold).astype(int)

    # Create the classification report
    class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, digits=4)
    class_report = {
        key: value for key, value in class_report.items() if key.isnumeric() and value['support'] > 0
    }

    true_labels = mlb_encoder.inverse_transform(y_true)
    pred_labels = mlb_encoder.inverse_transform(y_pred)
    true_labels_mt = np.zeros_like([*microthesaurus], dtype=np.int8)
    true_labels_domain = np.zeros_like([*domain], dtype=np.int8)
    pred_labels_mt = np.zeros_like([*microthesaurus], dtype=np.int8)
    pred_labels_domain = np.zeros_like([*domain], dtype=np.int8)

    # The true labels are split into MT and domain labels
    for labels in true_labels:
        for label in labels:
            if str(label) in mt_labels:
                if mt_labels[str(label)] in microthesaurus:
                    true_labels_mt[microthesaurus[mt_labels[str(label)]]] = 1
                if mt_labels[str(label)][:2] in domain:
                    true_labels_domain[domain[mt_labels[str(label)][:2]]] = 1

    # The predicted labels are split into MT and domain labels
    for labels in pred_labels:
        for label in labels:
            if str(label) in mt_labels:
                if mt_labels[str(label)] in microthesaurus:
                    pred_labels_mt[microthesaurus[mt_labels[str(label)]]] = 1
                if mt_labels[str(label)][:2] in domain:
                    pred_labels_domain[domain[mt_labels[str(label)][:2]]] = 1

    # Return all the metrics
    return {
        'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred, average = 'micro'),
        'precision': precision_score(y_true, y_pred, average = 'micro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average = 'micro', zero_division=0),
        'hamming': hamming_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'ndcg_1': ndcg_score(y_true, y_pred, k=1),
        'ndcg_3': ndcg_score(y_true, y_pred, k=3),
        'ndcg_5': ndcg_score(y_true, y_pred, k=5),
        'ndcg_10': ndcg_score(y_true, y_pred, k=10),
        'f1_micro_mt': f1_score(y_true=true_labels_mt, y_pred=pred_labels_mt, average='micro', zero_division=0),
        'f1_macro_mt': f1_score(y_true=true_labels_mt, y_pred=pred_labels_mt, average='macro', zero_division=0),
        'f1_weighted_mt': f1_score(y_true=true_labels_mt, y_pred=pred_labels_mt, average='weighted', zero_division=0),
        'f1_micro_domain': f1_score(y_true=true_labels_domain, y_pred=pred_labels_domain, average='micro', zero_division=0),
        'f1_macro_domain': f1_score(y_true=true_labels_domain, y_pred=pred_labels_domain, average='macro', zero_division=0),
        'f1_weighted_domain': f1_score(y_true=true_labels_domain, y_pred=pred_labels_domain, average='weighted', zero_division=0),
        'roc_auc_mt': roc_auc_score(true_labels_mt, pred_labels_mt, average = 'micro'),
        'roc_auc_domain': roc_auc_score(true_labels_domain, pred_labels_domain, average = 'micro'),
        'precision_mt': precision_score(true_labels_mt, pred_labels_mt, average = 'micro', zero_division=0),
        'precision_domain': precision_score(true_labels_domain, pred_labels_domain, average = 'micro', zero_division=0),
        'recall_mt': recall_score(true_labels_mt, pred_labels_mt, average = 'micro', zero_division=0),
        'recall_domain': recall_score(true_labels_domain, pred_labels_domain, average = 'micro', zero_division=0),
        'hamming_mt': hamming_loss(true_labels_mt, pred_labels_mt),
        'hamming_domain': hamming_loss(true_labels_domain, pred_labels_domain),
        'accuracy_mt': accuracy_score(true_labels_mt, pred_labels_mt),
        'accuracy_domain': accuracy_score(true_labels_domain, pred_labels_domain),
    }, class_report

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
        if lang == directory[:2]:
            for i in range(len(seeds)):
                if data_type == "train":
                    print("Loading training and dev data from directory {}...".format(os.path.join(data_path, directory, "split_{}".format(i))))

                    # The data is stored in numpy arrays, so it has to be converted to tensors.
                    train_X = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "train_X.npy")))
                    train_mask = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "train_mask.npy")))
                    train_y = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "train_y.npy"))).float()

                    assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

                    dev_X = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "dev_X.npy")))
                    dev_mask = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "dev_mask.npy")))
                    dev_y = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "dev_y.npy"))).float()

                    assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]

                    dataset_train = TensorDataset(train_X, train_mask, train_y)

                    dataset_dev = TensorDataset(dev_X, dev_mask, dev_y)
                    datasets.append((dataset_train, dataset_dev, train_y.shape[1]))

                elif data_type == "test":
                    print("Loading test data from directory {}...".format(os.path.join(data_path, directory, "split_{}".format(i))))
                    test_X = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "test_X.npy")))
                    test_mask = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "test_mask.npy")))
                    test_y = from_numpy(np.load(os.path.join(data_path, directory, "split_{}".format(i), "test_y.npy"))).float()

                    assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

                    dataset_test = TensorDataset(test_X, test_mask, test_y)

                    datasets.append((dataset_test))
            break

    return datasets