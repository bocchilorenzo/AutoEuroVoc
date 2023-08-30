import json
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from re import compile
import numpy as np
from tqdm import tqdm
import gzip
import json
from os import listdir, makedirs, path
from tqdm import tqdm
import pickle
import spacy
import argparse
from pagerange import PageRange

from text_summarizer import lex_utils
from text_summarizer import Cache

def summarize(args):
    print(f"Arguments: {args}")

    print(f"Reading seeds file {args.seed_path}")
    with open(args.seed_path, "r") as f:
        seed_list = json.load(f)

    makedirs(args.output_path, exist_ok=True)

    main_dir = args.data_path

    # Get all the files in the directory
    if args.years == "all":
        file_list = [year for year in listdir(main_dir)
                      if path.isfile(path.join(main_dir, year))
                      and year.endswith(".json.gz")]
    else:
        file_list = [str(year) + ".json.gz" for year in PageRange(args.years).pages]

    cache = None
    if args.cache:
        cache = Cache(args.cache)

    print(cache)

    print(f"Loading model for spaCy:", args.spacy_model)
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        print('Downloading language model for spaCy')
        spacy.cli.download(args.spacy_model)
        nlp = spacy.load(args.spacy_model)

    nlp.max_length = args.spacy_max_length

    for seed in seed_list:

        makedirs(path.join(args.output_path, seed), exist_ok=True)

        print(f"Getting information for seed {seed}")
        trainList = set(seed_list[seed]['train'])

        # Set up the variables
        if args.mode == "label":
            label_map = {}
        else:
            docs_to_process = []

        print("Reading files...")
        for file in tqdm(file_list):
            with gzip.open(path.join(main_dir, file), "rt", encoding="utf-8") as f:
                data = json.load(f)
                for doc in data:
                    if doc not in trainList:
                        continue

                    text = data[doc]["full_text"]

                    if cache:
                        s = cache.getFile(doc)
                        if s:
                            sentences = json.loads(s)
                        else:
                            sentences = lex_utils.get_data(text, nlp, args.spacy_num_threads)
                            cache.writeFile(docID, json.dumps(sentences))
                    else:
                        sentences = lex_utils.get_data(text, nlp, args.spacy_num_threads)

                    docLemmas = []
                    for sentence in sentences:
                        for i in range(len(sentence["token"])):
                            pos = sentence["pos"][i]
                            if pos in lex_utils.allowedPos:
                                docLemmas.append(sentence["lemma"][i])

                    if args.mode == "label":
                        # If we chose to summarize by label, we concatenate all the texts of the documents with the same label
                        for label in data[doc]["eurovoc_classifiers"]:
                            if label not in label_map:
                                label_map[label] = " ".join(docLemmas)
                            else:
                                label_map[label] = " ".join([label_map[label], " ".join(docLemmas)])
                    else:
                        # If we chose to summarize by document, we just append the text of the document to the list
                        docs_to_process.append(" ".join(docLemmas))

        del data, doc

        print("Creating TF-IDF matrix...")
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm="l2", smooth_idf=True)
        if args.mode == "label":
            tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(list(label_map.values()))
        else:
            tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs_to_process)
        feature_indices = tfidf_vectorizer.vocabulary_

        # Convert to Compressed Sparse Column format to allow for fast column slicing and indexing
        tfidf_vectorizer_vectors = tfidf_vectorizer_vectors.tocsc()

        # Delete the variables we don't need anymore to save up memory
        if args.mode == "label":
            del label_map
        else:
            del docs_to_process

        for file in file_list:
            print(f"Summarizing {file}...")
            with gzip.open(path.join(main_dir, file), "rt", encoding="utf-8") as f:
                data = json.load(f)
                for doc in tqdm(list(data.keys())):

                    text = data[doc]["full_text"]

                    if cache:
                        s = cache.getFile(doc)
                        if s:
                            print("Load from cache")
                            sentences = json.loads(s)
                        else:
                            sentences = lex_utils.get_data(text, nlp, args.spacy_num_threads)
                            cache.writeFile(docID, json.dumps(sentences))
                    else:
                        sentences = lex_utils.get_data(text, nlp, args.spacy_num_threads)
                    
                    data[doc]["full_text"] = [sentence['text'] for sentence in sentences]

                    sent_eval = []
                    for sentence in sentences:

                        sentLemmas = []
                        for i in range(len(sentence["token"])):
                            pos = sentence["pos"][i]
                            if pos in lex_utils.allowedPos:
                                sentLemmas.append(sentence["lemma"][i])

                        sent_score = []
                        for word in sentLemmas:
                            # For each word in the sentence, get the TFIDF score as the maximum the word has in the matrix
                            if word in feature_indices:
                                sent_score.append(tfidf_vectorizer_vectors[:, feature_indices[word]].max())
                            else:
                                sent_score.append(0)
                        # Where the score is 0, replace it with NaN so it does not influence the mean of the sentence
                        sent_score = [np.nan if i == 0 else i for i in sent_score]
                        sent_score = np.array(sent_score, dtype=np.float64)
                        if args.scoring == "max":
                            to_append = np.nanmax(sent_score, initial=0)
                        else:
                            to_append = np.nanmean(sent_score)
                        sent_eval.append(to_append if not np.isnan(to_append) else 0)
                    data[doc]["importance"] = sent_eval
            with gzip.open(path.join(args.output_path, seed, file.replace(".json.gz", "_sum_tfidf_l2.json.gz")), "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        if args.save_vectorizers:
            print("Saving vectorizers...")
            with open(path.join(args.output_path, seed, "tfidf_vectorizer.pkl"), "wb") as f:
                pickle.dump(tfidf_vectorizer, f)
            with open(path.join(args.output_path, seed, "tfidf_vectorizer_vectors.pkl"), "wb") as f:
                pickle.dump(tfidf_vectorizer_vectors, f)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFIDF-based summarizer for the dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./data/it/extracted/few_labels_removed", help="Directory containing the input dataset")
    parser.add_argument("--output_path", type=str, default="./data-summ/it", help="Directory containing the output dataset")

    parser.add_argument("--spacy_model", metavar="NAME", type=str, default="en_core_web_sm", help="spaCy model name")
    parser.add_argument("--spacy_max_length", metavar="LENGTH", type=int, default=1000000, help="Maximum length to pass to spaCy (leave it to the default value if you don't have issues with memory)")
    parser.add_argument("--spacy_num_threads", metavar="NUM", type=int, default=8, help="Number of processes for spaCy")

    parser.add_argument("--seed_path", type=str, help="JSON file for seeds", required=True)

    parser.add_argument("--years", type=str, default="all", help="Range of years to summarize (e.g. 2010-2022 includes 2022). Use 'all' to process all the files in the given folder.")
    parser.add_argument("--cache", type=str, default=None, help="Cache folder for tokenization results")

    parser.add_argument("--scoring", type=str, default="max", choices=["max", "mean"], help="Scoring method for the TFIDF vectors")
    parser.add_argument("--mode", type=str, default="label", choices=["document", "label"], help="Calculate the TFIDF score for the whole document or for each label")
    parser.add_argument("--save_vectorizers", action="store_true", default=False, help="Save the TFIDF vectorizers")
    args = parser.parse_args()

    summarize(args)