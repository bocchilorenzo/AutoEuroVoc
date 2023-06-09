# based on pyeurovoc's preprocess.py

import argparse
import yaml
import os
import re
from transformers import AutoTokenizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from torch import tensor, ones_like, device, multiprocessing
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import gzip
import pickle
import math
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool

seeds = []

# The domains and microthesaurus labels are loaded from the json files
with open("config/domain_labels_position.json", "r") as fp:
    domain = json.load(fp)
with open("config/mt_labels_position.json", "r") as fp:
    microthesaurus = json.load(fp)
with open("config/mt_labels.json", "r", encoding="utf-8") as file:
    mt_labels = json.load(file)

def save_splits(X, masks, y, directory, mlb):
    """
    Save the splits of the dataset.
    
    :param X: List of inputs.
    :param masks: List of masks.
    :param y: List of labels.
    :param directory: Language directory.
    :param mlb: MultiLabelBinarizer object.
    """
    global seeds

    print(f"{datetime.now().replace(microsecond=0)} - Saving splits...")

    for i, seed in enumerate(seeds):
        np.random.seed(int(seed))
        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.2, 0.8])
        train_idx, aux_idx = next(stratifier.split(X, y))
        train_X, train_mask, train_y = X[train_idx, :], masks[train_idx, :], y[train_idx, :]

        assert train_X.shape[0] == train_mask.shape[0] == train_y.shape[0]

        stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.5, 0.5])
        dev_idx, test_idx = next(stratifier.split(X[aux_idx, :], y[aux_idx, :]))
        dev_X, dev_mask, dev_y = X[dev_idx, :], masks[dev_idx, :], y[dev_idx, :]
        test_X, test_mask, test_y = X[test_idx, :], masks[test_idx, :], y[test_idx, :]

        assert dev_X.shape[0] == dev_mask.shape[0] == dev_y.shape[0]
        assert test_X.shape[0] == test_mask.shape[0] == test_y.shape[0]

        to_print = f"{i} - Splitted the documents in - train: {train_X.shape[0]}, dev: {dev_X.shape[0]}, test: {test_X.shape[0]}"
        print(to_print)

        with open(os.path.join(args.data_path, directory, "stats.txt"), "a+") as f:
            f.write(to_print + "\n")

        if not os.path.exists(os.path.join(args.data_path, directory, f"split_{i}")):
            os.makedirs(os.path.join(args.data_path, directory, f"split_{i}"))

        np.save(os.path.join(args.data_path, directory, f"split_{i}", "train_X.npy"), train_X)
        np.save(os.path.join(args.data_path, directory, f"split_{i}", "train_mask.npy"), train_mask)
        np.save(os.path.join(args.data_path, directory, f"split_{i}", "train_y.npy"), train_y)

        np.save(os.path.join(args.data_path, directory, f"split_{i}", "dev_X.npy"), dev_X)
        np.save(os.path.join(args.data_path, directory, f"split_{i}", "dev_mask.npy"), dev_mask)
        np.save(os.path.join(args.data_path, directory, f"split_{i}", "dev_y.npy"), dev_y)

        np.save(os.path.join(args.data_path, directory, f"split_{i}", "test_X.npy"), test_X)
        np.save(os.path.join(args.data_path, directory, f"split_{i}", "test_mask.npy"), test_mask)
        np.save(os.path.join(args.data_path, directory, f"split_{i}", "test_y.npy"), test_y)

        sample_labs = mlb.inverse_transform(train_y)
        labs_count = {"total_samples": len(sample_labs), "labels": {label: 0 for label in mlb.classes_}}

        for sample in sample_labs:
            for label in sample:
                labs_count["labels"][label] += 1
        
        with open(os.path.join(args.data_path, directory, f"split_{i}", "train_labs_count.json"), "w") as fp:
            json.dump(labs_count, fp)

        X, masks, y = shuffle(X, masks, y, random_state=int(seed))

def process_year(params):
    """
    Process a year of the dataset.

    :param params: Tuple containing, the path to the year, the tokenizer name and the arguments.
    :return: List of inputs, masks and labels.
    """
    document_ct = 0
    big_document_ct = 0
    unk_ct = 0
    tokens_ct = 0

    path, tokenizer_name, args = params

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": args.max_length}

    list_inputs = []
    list_labels = []
    list_masks = []

    if args.multi_core:
        print(f"{datetime.now().replace(microsecond=0)} - Processing {path}")

    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
        j = 1
        for doc in data:
            if not args.multi_core:
                print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
            j += 1
            text = ""
            if args.add_mt_do:
                # Add MT and DO labels
                labels = set(data[doc]["eurovoc_classifiers"]) if "eurovoc_classifiers" in data[doc] else set(data[doc]["eurovoc"])
                to_add = set()
                for label in labels:
                    if label in mt_labels:
                        if mt_labels[label] in microthesaurus:
                            to_add.add(mt_labels[label] + "_mt")
                        if mt_labels[label][:2] in domain:
                            to_add.add(mt_labels[label][:2] + "_do")
                
                labels = list(labels.union(to_add))
            else:
                labels = data[doc]["eurovoc_classifiers"] if "eurovoc_classifiers" in data[doc] else data[doc]["eurovoc"]

            if args.title_only:
                text = data[doc]["title"]
            else:
                if args.add_title:
                    text = data[doc]["title"] + " "
                
                if args.summarized:
                    full_text = data[doc]["full_text"]
                    phrase_importance = []
                    i = 0

                    for imp in data[doc]["importance"]:
                        if not math.isnan(imp):
                            phrase_importance.append((i, imp))
                        i += 1
                    
                    phrase_importance = sorted(phrase_importance, key=lambda x: x[1], reverse=True)

                    # First, we get the most important phrases until the maximum length is reached.
                    if len(" ".join([full_text[phrase[0]] for phrase in phrase_importance]).split()) > args.max_length:
                        backup = deepcopy(phrase_importance)
                        while len(" ".join([full_text[phrase[0]] for phrase in phrase_importance]).split()) > args.max_length:
                            phrase_importance = phrase_importance[:-1]
                        phrase_importance.append(backup[len(phrase_importance)])

                    # Then, we sort the phrases by their position in the document.
                    phrase_importance = sorted(phrase_importance, key=lambda x: x[0])
                    text += " ".join([full_text[phrase[0]] for phrase in phrase_importance])
                else:
                    text += data[doc]["full_text"] if "full_text" in data[doc] else data[doc]["text"]
            
            text = re.sub(r'\r', '', text)
            
            # The following replacement is not necessary in the other datasets because it was already done.
            if "senato" in path:
                text = re.sub(r'\n', ' ', text)
                text = re.sub(" +", " ", text).strip()
            
            if args.limit_tokenizer:
                inputs_ids = tensor(tokenizer.encode(text, **tokenizer_kwargs))
            else:
                inputs_ids = tensor(tokenizer.encode(text))

            if not args.limit_tokenizer:
                document_ct += 1

                # We count the number of unknown tokens and the total number of tokens.
                for token in inputs_ids[1: -1]:
                    if token == tokenizer.unk_token_id:
                        unk_ct += 1

                    tokens_ct += 1

                # If the input is over the maximum length, we cut it and increment the count of big documents.
                if len(inputs_ids) > args.max_length:
                    big_document_ct += 1
                    inputs_ids = inputs_ids[:args.max_length]

            list_inputs.append(inputs_ids)
            list_labels.append(labels)
            list_masks.append(ones_like(inputs_ids))
    
    del data, inputs_ids, labels, text, tokenizer

    if len(list_inputs) == 0:
        print("No documents found in the dataset.")
        to_print = ""
    else:
        if not args.limit_tokenizer:
            if not args.multi_core:
                to_print = f"Dataset stats: - total documents: {document_ct}, big documents: {big_document_ct}, ratio: {big_document_ct / document_ct * 100:.4f}%"
            else:
                to_print = f"Stats for {path}: - total documents: {document_ct}, big documents: {big_document_ct}, ratio: {big_document_ct / document_ct * 100:.4f}%"
            to_print += f"\n               - total tokens: {tokens_ct}, unk tokens: {unk_ct}, ratio: {unk_ct / tokens_ct * 100:.4f}%"
            print(to_print)
        else:
            to_print = ""

    return list_inputs, list_masks, list_labels, to_print

def process_datasets(data_path, directory, tokenizer_name):
    """
    Process the datasets and save them in the specified directory.

    :param data_path: Path to the data.
    :param directory: Language directory.
    :param tokenizer_name: Name of the tokenizer to use.
    """

    list_inputs = []
    list_masks = []
    list_labels = []
    list_stats = []
    list_years = []

    # If no years are specified, process all the downloaded years depending on the arguments.
    if args.years == "all":
        if not args.summarized:
            args.years = ",".join([year.split(".")[0] for year in os.listdir(os.path.join(data_path, directory)) if "sum" not in year and os.path.isfile(os.path.join(data_path, directory, year)) and year.endswith(".json.gz")])
        else:
            args.years = ",".join([year.split(".")[0] for year in os.listdir(os.path.join(data_path, directory)) if os.path.isfile(os.path.join(data_path, directory, year)) and year.endswith(".json.gz") and args.summ_mode in year])
    else:
        if "," not in args.years:
            args.years += "," + args.years
        
        if not args.summarized:
            args.years = ",".join([str(year) for year in range(int(args.years.split(",")[0]), int(args.years.split(",")[1]) + 1)])
        else:
            args.years = ",".join([str(year) + "_sum_" + args.summ_mode for year in range(int(args.years.split(",")[0]), int(args.years.split(",")[1]) + 1)])
    
    print(f"Years to process: '{args.years}'\n")

    # If the dataset is the Senato one, there is only one file to process.
    if directory == "senato":
        print("Processing Senato dataset...")
        list_inputs, list_masks, list_labels, year_stats = process_year((os.path.join(data_path, directory, "aic-out.json.gz"), tokenizer_name, args))
    else:
        if not args.multi_core:
            for year in args.years.split(","):
                if args.summarized:
                    print(f"Processing summarized year: '{year}'...")
                else:
                    print(f"Processing year: '{year}'...")
                year_inputs, year_masks, year_labels, year_stats = process_year((os.path.join(data_path, directory, f"{year}.json.gz"), tokenizer_name, args))
                
                list_inputs += year_inputs
                list_masks += year_masks
                list_labels += year_labels
                list_stats.append(year_stats)
                list_years.append(year)
        else:
            print("Processing data in parallel...")

            with Pool(args.cpu_count) as p:
                results = list(
                    p.imap(process_year, [(os.path.join(data_path, directory, f"{year}.json.gz"), tokenizer_name, args) for year in args.years.split(",")])
                )
            
            print(f"{datetime.now().replace(microsecond=0)} - Merging results...")
            for i, result in enumerate(results):
                list_inputs += result[0]
                list_masks += result[1]
                list_labels += result[2]
                list_stats.append(result[3])
                list_years.append(args.years.split(",")[i])
            
            del results

    assert len(list_inputs) == len(list_masks) == len(list_labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(list_labels)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    X = pad_sequence(list_inputs, batch_first=True, padding_value=tokenizer.pad_token_id).numpy()
    masks = pad_sequence(list_masks, batch_first=True, padding_value=0).numpy()

    with open(os.path.join(args.data_path, directory, "mlb_encoder.pickle"), "wb") as pickle_fp:
        pickle.dump(mlb, pickle_fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not args.limit_tokenizer:
        with open(os.path.join(args.data_path, directory, "stats.txt"), "w") as stats_fp:
            for year, year_stats in zip(list_years, list_stats):
                stats_fp.write(f"Year: {year}\n{year_stats}\n\n")

    save_splits(X, masks, y, directory, mlb)

def preprocess_data():
    """
    Load the configuration file and process the data.
    """
    with open("config/models.yml", "r") as fp:
        config = yaml.safe_load(fp)
    
    global seeds
    with open("config/seeds.txt", "r") as fp:
        seeds = fp.read().splitlines()

    print(f"Tokenizers config:\n{format(config)}")
    
    if args.senato:
        print(f"\nWorking on Senato data...")
        lang = "it"
        print(f"Lang: '{lang}', Tokenizer: '{config[lang]}'")
        process_datasets(args.data_path, "senato", config[lang])
    else:
        for directory in os.listdir(args.data_path):
            # If we specified one or more languages, we only process those.
            if args.langs != "all" and directory not in args.langs.split(","):
                continue
            
            print(f"\nWorking on directory: {format(directory)}...")
            lang = directory
            print(f"Lang: '{lang}', Tokenizer: '{config[lang]}'")

            process_datasets(args.data_path, directory, config[lang])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="data/", help="Path to the data to process.")
    parser.add_argument("--years", type=str, default="all", help="Year range to be processed, separated by a comma (e.g. 2010,2020 will get all the years between 2010 and 2020 included). Write 'all' to process all the years.")
    parser.add_argument("--langs", type=str, default="it", help="Languages to be processed, separated by a comme (e.g. en,it). Write 'all' to process all the languages.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum number of words of the text to be processed.")
    parser.add_argument("--limit_tokenizer", action="store_true", default=False, help="Limit the tokenizer length to the maximum number of words. This will remove the statistics for the documents length.")
    parser.add_argument("--add_title", action="store_true", default=False, help="Add the title to the text.")
    parser.add_argument("--title_only", action="store_true", default=False, help="Use only the title as input.")
    parser.add_argument("--add_mt_do", action="store_true", default=False, help="Add the MicroThesaurus and Domain labels to be predicted.")
    parser.add_argument("--senato", action="store_true", default=False, help="Process the Senato data instead of the EUR-Lex one.")
    parser.add_argument("--summarized", action="store_true", default=False, help="Process the summarized data instead of the full text one.")
    parser.add_argument("--summ_mode", type=str, default="centroid_full", choices=["centroid_full", "centroid_compressed", "centroid_compressed_bigram", "tfidf_l1", "tfidf_l2"], help="Summarization method to use. Only used if --summarized is also used.")
    parser.add_argument("--multi_core", action="store_true", default=False, help="Use multiple cores to process the data.")
    parser.add_argument("--cpu_count", type=int, default=2, help="Number of cores to use. Only used if --multi_core is also used.")
    args = parser.parse_args()

    preprocess_data()