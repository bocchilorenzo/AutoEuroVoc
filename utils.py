from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, hamming_loss, ndcg_score, precision_score, recall_score, jaccard_score, matthews_corrcoef
from torch import sigmoid, Tensor, stack, from_numpy
import os
import numpy as np
from torch.utils.data import TensorDataset

def sklearn_metrics(y_true, predictions, threshold=0.5):
    """
    Return the metrics and classification report for the predictions.
    
    :param y_true: True labels.
    :param predictions: Predictions.
    :param threshold: Threshold for the predictions. Default: 0.5.
    :return: A dictionary with the metrics and a classification report.
    """
    # Convert the predictions to binary
    probs = sigmoid(Tensor(predictions))
    y_pred = (probs.detach().numpy() >= threshold).astype(int)

    # Create the classification report
    class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True, digits=4)
    class_report = {
        key: value for key, value in class_report.items() if key.isnumeric() and value['support'] > 0
    }

    references = np.array(y_true)
    predictions = np.array(y_pred)

    matthews_corr = [
        matthews_corrcoef(predictions[:, i], references[:, i], sample_weight=None)
        for i in range(references.shape[1])
    ]
    
    # Return all the metrics
    return {
        'f1_micro': f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0),
        'f1_samples': f1_score(y_true=y_true, y_pred=y_pred, average='samples', zero_division=0),
        'jaccard': jaccard_score(y_true, y_pred, average = 'micro', zero_division=0),
        'matthews_macro': np.mean(matthews_corr),
        'roc_auc': roc_auc_score(y_true, y_pred, average = 'micro'),
        'precision': precision_score(y_true, y_pred, average = 'micro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average = 'micro', zero_division=0),
        'hamming': hamming_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'ndcg_1': ndcg_score(y_true, y_pred, k=1),
        'ndcg_3': ndcg_score(y_true, y_pred, k=3),
        'ndcg_5': ndcg_score(y_true, y_pred, k=5),
        'ndcg_10': ndcg_score(y_true, y_pred, k=10),
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