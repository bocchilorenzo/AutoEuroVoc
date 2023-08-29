from os import makedirs, path, remove, rename
from text_summarizer import Summarizer
import json, gzip, argparse, zipfile, urllib.request
from tqdm import tqdm
from pagerange import PageRange
from os import listdir, path
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# def check_models():
#     if not path.exists("./text_summarizer/models"):
#         print("Downloading udpipe models...")
#         url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/udpipe-ud-2.5-191206.zip?sequence=1&isAllowed=y"
#         urllib.request.urlretrieve(url, path.join("./", "udpipe.zip"))
#         with zipfile.ZipFile(path.join("./", "udpipe.zip"), 'r') as zip_ref:
#             zip_ref.extractall(path.join("./", "text_summarizer"))
#         rename(path.join("./", "text_summarizer", "udpipe-ud-2.5-191206"), path.join("./", "text_summarizer", "models"))

#         remove(path.join("./", "udpipe.zip"))

def summarize_data(args):
    # if args.tokenizer == "udpipe1":
    #     check_models()

    print(f"Loading {args.model_type} model...")
    model = Summarizer(
        language=args.summ_lang,
        model_path=args.model_path,
        compressed=args.compressed,

        tokenizer=args.tokenizer,
        udpipe_url=args.udpipe_url,
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

    if args.years == "all":
        years = [year for year in listdir(path_initial)
                      if path.isfile(path.join(path_initial, year))
                      and year.endswith(".json.gz")]
    else:
        years = [str(year) + ".json.gz" for year in PageRange(args.years).pages]

    print(f"Working on data from {path_initial}. Language: {args.summ_lang}. Files to process: {', '.join(years)}.")

    for file in years:
        try:
            with gzip.open(path.join(path_initial, file), "rt", encoding="utf-8") as fp:
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
    parser.add_argument("--summ_lang", metavar="LANG", type=str, default="italian", help="Language of the summarizer model")
    parser.add_argument("--model_path", metavar="PATH", type=str, default="./cc.it.300.bin", help="Path to the folder containing the summarizer model")
    parser.add_argument("--compressed", action="store_true", default=False, help="Whether the model is compressed or not")

    parser.add_argument("--tokenizer", type=str, default="nltk", choices=["udpipe", "nltk", "spacy"], help="Tokenizer to use for the summarizer. NOTE: right now spacy is only available for english texts")
    parser.add_argument("--udpipe_url", metavar="URL", type=str, default="http://127.0.0.1:30101/process", help="UDPipe URL")
    parser.add_argument("--spacy_model", metavar="NAME", type=str, default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--spacy_max_length", metavar="LENGTH", type=int, default=1000000, help="Maximum length to pass to spaCy (leave it to the default value if you don't have issues with memory)")
    parser.add_argument("--spacy_num_threads", metavar="NUM", type=int, default=8, help="Number of processes for spaCy")

    parser.add_argument("--min_sent_length", metavar="NUM", type=int, default=20, help="Minimum length of a sentence (in characters, 0 means no limit)")
    parser.add_argument("--max_sent_length", metavar="NUM", type=int, default=300, help="Minimum length of a sentence (in characters, 0 means no limit)")
    parser.add_argument("--max_sent_entity_ratio", metavar="NUM", type=float, default=0.5, help="Max ratio entity-tokens/tokens in a sentence")

    parser.add_argument("--model_type", type=str, default="fasttext", choices=["fasttext", "word2vec"], help="Type of the summarizer model")
    parser.add_argument("--cache", type=str, default=None, help="Cache folder for tokenization results")
    parser.add_argument("--lang", type=str, default="italian", help="Language of the summarizer model")
    parser.add_argument("--years", type=str, default="all", help="Range of years to summarize (e.g. 2010-2022 includes 2022). Use 'all' to process all the files in the given folder.")

    args = parser.parse_args()
    summarize_data(args)