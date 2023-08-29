from os import makedirs, path, remove, rename
from text_summarizer import Summarizer
import json, gzip, argparse, zipfile, urllib.request
from tqdm import tqdm
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def summarize_data(args):

    print(f"Loading {args.model_type} model...")
    model = Summarizer(
        model_path=args.model_path,
        compressed=args.compressed,

        spacy_model=args.spacy_model,
        spacy_max_length=args.spacy_max_length,
        spacy_num_threads=args.spacy_num_threads,
        
        model_type=args.model_type,
        cache=args.cache,
        min_sent_length=args.min_sent_length,
        max_sent_length=args.max_sent_length,
        max_sent_entity_ratio=args.max_sent_entity_ratio,
    )

    path_initial = args.data_path
    new_path = args.output_path
    makedirs(new_path, exist_ok=True)

    years = [str(i) for i in range(int(args.years.split("-")[0]), int(args.years.split("-")[1]) + 1)]

    print(f"Working on data from {path_initial}. Years to process: {years}.")

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
            ids, text = model.summarize(text, doc)
            if len(ids) == 0 and len(text) == 0:
                to_del.add(doc)
                continue
            data[doc]["full_text"] = text
            data[doc]["importance"] = [id_imp[1] for id_imp in ids]
        
        print(f"Documents removed: {len(to_del)}")
        if len(to_del) > 0:
            print(f"List of removed documents: {', '.join(to_del)}")
        for doc in to_del:
            del data[doc]
        with gzip.open(path.join(new_path, f"{file}_sum_centroid{'_compressed' if args.compressed else '_full'}.json.gz"), "wt", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", metavar="FOLDER", type=str, default="./", help="Path to the folder containing the json.gz files")
    parser.add_argument("--output_path", metavar="FOLDER", type=str, default="./summarized", help="Path to the folder where the summarized files will be saved")
    parser.add_argument("--model_path", metavar="PATH", type=str, default="./cc.it.300.bin", help="Path to the folder containing the summarizer model")
    parser.add_argument("--compressed", action="store_true", default=False, help="Whether the model is compressed or not")

    parser.add_argument("--spacy_model", metavar="NAME", type=str, default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--spacy_max_length", metavar="LENGTH", type=int, default=1000000, help="Maximum length to pass to spaCy (leave it to the default value if you don't have issues with memory)")
    parser.add_argument("--spacy_num_threads", metavar="NUM", type=int, default=8, help="Number of processes for spaCy")

    parser.add_argument("--min_sent_length", metavar="NUM", type=int, default=20, help="Minimum length of a sentence (in characters, 0 means no limit)")
    parser.add_argument("--max_sent_length", metavar="NUM", type=int, default=500, help="Minimum length of a sentence (in characters, 0 means no limit)")
    parser.add_argument("--max_sent_entity_ratio", metavar="NUM", type=float, default=0.5, help="Max ratio entity-tokens/tokens in a sentence")

    parser.add_argument("--model_type", type=str, default="fasttext", choices=["fasttext", "word2vec"], help="Type of the summarizer model")
    parser.add_argument("--years", type=str, default="2010-2022", help="Range of years to summarize (extremes included)")
    parser.add_argument("--cache", type=str, default=None, help="Cache folder for tokenization results")

    args = parser.parse_args()
    summarize_data(args)