import argparse
import yaml
import os
import re
import numpy as np
import json
import gzip
import pickle
import math
from datetime import datetime
from transformers import AutoTokenizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from torch import tensor, ones_like
from torch.nn.utils.rnn import pad_sequence

from scripts.create_dataset.text_summarizer.lex_utils import get_years, init_procedure, add_argument

script_folder = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(script_folder, "config/models.yml"), "r") as fp:
    config = yaml.safe_load(fp)

def process_year_data(path, tokenizer_name, args):
    """
    Process a year of the dataset.

    :param path: Path to the data.
    :param tokenizer_name: Name of the tokenizer to use.
    :param args: Command line arguments.
    :return: List of inputs, masks and labels.
    """
    document_ct = 0
    big_document_ct = 0
    unk_ct = 0
    tokens_ct = 0

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": args.max_length}

    # list_inputs = []
    # list_labels = []
    # list_masks = []

    outData = {}

    with gzip.open(path, "rt", encoding="utf-8") as file:
        data = json.load(file)

        summarize = False

        j = 1
        # Process the text
        for doc in data:
            if not summarize and "importance" in data[doc]:
                print("This file is summarized")
                summarize = True

            print(f"{datetime.now().replace(microsecond=0)} - {j}/{len(data)}", end="\r")
            j += 1
            # if j > 10:
            #     break
            outData[doc] = {}
            text = ""
            labels = data[doc]["eurovoc_classifiers"] if "eurovoc_classifiers" in data[doc] else data[doc]["eurovoc"]

            if args.title_only:
                text = data[doc]["title"]
            else:
                if not args.skip_title:
                    text = data[doc]["title"] + " "
                
                if "summarized_text" in data[doc]:
                    text += data[doc]["summarized_text"]
                else:
                    if summarize:
                        full_text = data[doc]["full_text"]
                        phrase_importance = []
                        i = 0

                        for imp in data[doc]["importance"]:
                            if math.isnan(imp):
                                imp = 0
                            phrase_importance.append((i, imp))
                            i += 1
                        
                        phrase_importance = sorted(phrase_importance, key=lambda x: x[1], reverse=True)

                        new_text = []
                        phrase_index = 0
                        while len(" ".join([full_text[phrase[0]] for phrase in new_text]).split()) < args.max_length and phrase_index < len(phrase_importance):
                            new_text.append(phrase_importance[phrase_index])
                            phrase_index += 1

                        # Then, we sort the phrases by their position in the document.
                        if len(new_text) > 0:
                            new_text = sorted(new_text, key=lambda x: x[0])
                            text += " ".join([full_text[phrase[0]] for phrase in new_text])
                        else:
                            text += " ".join([full_text[phrase[0]] for phrase in phrase_importance])
                    else:
                        text += data[doc]["full_text"] if "full_text" in data[doc] else data[doc]["text"]
            
            text = re.sub(r'\r', '', text)
            
            if args.limit_tokenizer:
                # Here, the text is cut to the maximum length before being tokenized,
                # potentially speeding up the process for long documents.
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

            outData[doc]['x'] = inputs_ids
            outData[doc]['y'] = labels
            outData[doc]['m'] = ones_like(inputs_ids)
    
    print(f"Finished at: {datetime.now().replace(microsecond=0)}")

    del data, inputs_ids, labels, tokenizer

    # Just some stats to print and save later.
    if len(outData) == 0:
        print("No documents found in the dataset.")
        to_print = ""
    else:
        if not args.limit_tokenizer:
            to_print = f"Dataset stats: - total documents: {document_ct}, big documents: {big_document_ct}, ratio: {big_document_ct / document_ct * 100:.4f}%"
            to_print += f"\n               - total tokens: {tokens_ct}, unk tokens: {unk_ct}, ratio: {unk_ct / tokens_ct * 100:.4f}%"
            print(to_print)
        else:
            to_print = ""

    return outData, to_print

def process_data_seeds(args):

    print(f"Tokenizers config:\n{format(config)}")

    years = init_procedure(args.data_path, args.output_path, args.years)

    if not os.path.exists(args.seed_file):
        print(f"File {seedsFile} does not exist, exiting")
        exit()

    seeds = []
    if args.seeds:
        seeds = args.seeds.split(",")

    seed_data = {}
    with open(args.seed_file, "r") as f:
        seed_data = json.load(f)

    tokenizer_name = config[args.lang]
    print(f"Tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    list_years = []
    list_stats = []

    allData = {}
    for (year, yearFile) in years.items():
        print(f"Processing year {year}")
        data, to_print = process_year_data(os.path.join(args.data_path, yearFile), tokenizer_name, args)

        allData.update(data)
        list_years.append(year[1])
        list_stats.append(to_print)

    statsFile = os.path.join(args.output_path, f"stats_{''.join(seeds)}.txt")

    if not args.limit_tokenizer:
        with open(statsFile, "w") as stats_fp:
            for year, year_stats in zip(list_years, list_stats):
                stats_fp.write(f"Year: {year}\n{year_stats}\n\n")

    for seed in seed_data:
        if seeds and seed not in seeds:
            continue
            
        print(f"Saving seed: {seed}")

        seedFolder = os.path.join(args.output_path, f"split_{seed}")
        if not os.path.exists(seedFolder):
            os.makedirs(seedFolder)

        totals = {}
        thisData = {"x": [], "y": [], "m": []}
        for l in seed_data[seed]:
            totals[l] = 0

            # l can be train, test, dev

            for docID in seed_data[seed][l]:
                if docID not in allData:
                    # print(f"Missing document {docID}")
                    continue
                totals[l] += 1
                thisData["x"].append(allData[docID]["x"])
                thisData["y"].append(allData[docID]["y"])
                thisData["m"].append(allData[docID]["m"])

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(thisData["y"])
        X = pad_sequence(thisData["x"], batch_first=True, padding_value=tokenizer.pad_token_id).numpy()
        masks = pad_sequence(thisData["m"], batch_first=True, padding_value=0).numpy()

        # Save the MultiLabelBinarizer.
        with open(os.path.join(seedFolder, "mlb_encoder.pickle"), "wb") as pickle_fp:
            pickle.dump(mlb, pickle_fp, protocol=pickle.HIGHEST_PROTOCOL)

        # print(X.shape)
        # print(y.shape)
        # print(masks.shape)
        # print(totals)

        start = 0
        for l in seed_data[seed]:

            this_X = X[start:start + totals[l], :]
            this_y = y[start:start + totals[l], :]
            this_m = masks[start:start + totals[l], :]
            start = totals[l]

            # print(this_X.shape)
            # print(this_y.shape)
            # print(this_m.shape)

            to_print = f"{seed} - {l}: {this_X.shape[0]}"
            print(to_print)

            with open(os.path.join(seedFolder, "stats.txt"), "a+") as f:
                f.write(to_print + "\n")

            # Save the splits
            np.save(os.path.join(seedFolder, f"{l}_X.npy"), this_X)
            np.save(os.path.join(seedFolder, f"{l}_mask.npy"), this_m)
            np.save(os.path.join(seedFolder, f"{l}_y.npy"), this_y)

            if l == "train":
                # Save the counts of each label, useful for weighted loss
                sample_labs = mlb.inverse_transform(this_y)
                labs_count = {"total_samples": len(sample_labs), "labels": {label: 0 for label in mlb.classes_}}

                for sample in sample_labs:
                    for label in sample:
                        labs_count["labels"][label] += 1
                
                with open(os.path.join(seedFolder, "train_labs_count.json"), "w") as fp:
                    json.dump(labs_count, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, default="en", help="Language to be processed (only for loading BERT model)")
    add_argument(parser, "years")
    add_argument(parser, "data_path")
    add_argument(parser, "output_path")
    add_argument(parser, "seed_file")
    add_argument(parser, "seeds")

    parser.add_argument("--skip_title", action="store_true", default=False, help="Do not add the title to the text.")
    parser.add_argument("--title_only", action="store_true", default=False, help="Use only the title as input.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum number of words of the text to be processed.")
    parser.add_argument("--limit_tokenizer", action="store_true", default=False, help="Limit the tokenizer length to the maximum number of words. This will remove the statistics for the documents length.")

    args = parser.parse_args()
    process_data_seeds(args)
