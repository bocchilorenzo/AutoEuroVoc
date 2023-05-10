from os import makedirs, path
from text_summarizer import Summarizer

import numpy as np
import json
import gzip
from tqdm import tqdm
import argparse
np.seterr(divide='ignore', invalid='ignore')

def summarize_data(args):
    print("Loading model...")
    model = Summarizer(
        language=args.summ_lang,
        model_path=args.model_path,
        compressed=args.compressed,
        tokenizer=args.tokenizer,
        model_type=args.model_type
    )

    path_initial = args.data_path
    new_path = args.output_path
    makedirs(new_path, exist_ok=True)

    years = [str(i) for i in range(int(args.years.split("-")[0]), int(args.years.split("-")[1]) + 1)]

    print(f"Working on data from {path_initial}. Language: {args.summ_lang}. Years to process: {years}.")

    for file in years:
        try:
            with gzip.open(path.join(path_initial, file+".json.gz"), "rt", encoding="utf-8") as fp:
                data = json.load(fp)
        except:
            print(f"Archive for {file} not found")
            continue
        to_del = set()
        for doc in tqdm(data, desc=file):
            text = data[doc]["full_text"]
            ids, text = model.summarize(text)
            if len(ids) == 0 and len(text) == 0:
                print(file, doc)
                to_del.add(doc)
                continue
            data[doc]["full_text"] = text
            data[doc]["importance"] = [id_imp[1] for id_imp in ids]
        
        for doc in to_del:
            del data[doc]
        with gzip.open(path.join(new_path, f"{file}_sum_centroid{'_compressed' if args.compressed else '_full'}.json.gz"), "wt", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./", help="Path to the folder containing the json.gz files")
    parser.add_argument("--output_path", type=str, default="./summarized", help="Path to the folder where the summarized files will be saved")
    parser.add_argument("--summ_lang", type=str, default="italian", help="Language of the summarizer model")
    parser.add_argument("--model_path", type=str, default="./cc.it.300.bin", help="Path to the folder containing the summarizer model")
    parser.add_argument("--compressed", action="store_true", default=False, help="Whether the model is compressed or not")
    parser.add_argument("--tokenizer", type=str, default="udpipe", choices=["udpipe", "nltk"], help="Tokenizer to use for the summarizer")
    parser.add_argument("--model_type", type=str, default="fasttext", choices=["fasttext", "word2vec"], help="Type of the summarizer model")
    parser.add_argument("--years", type=str, default="2010-2022", help="Range of years to summarize (extremes included)")

    args = parser.parse_args()
    summarize_data(args)