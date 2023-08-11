import json
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from re import compile
from nltk import download
download("punkt")
download("stopwords")
import numpy as np
from tqdm import tqdm
import gzip
import json
from os import listdir, makedirs, path
from tqdm import tqdm
import pickle
import spacy
import argparse

parser = argparse.ArgumentParser(description="TFIDF-based summarizer for the dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dir", type=str, default="../dataset/final/few_labels_removed", help="Directory containing the input dataset")
parser.add_argument("--out_dir", type=str, default="tfidf_label_mean", help="Directory containing the output dataset")
parser.add_argument("--mode", type=str, default="label", choices=["document", "label"], help="Calculate the TFIDF score for the whole document or for each label")
parser.add_argument("--scoring", type=str, default="max", choices=["max", "mean"], help="Scoring method for the TFIDF vectors")
parser.add_argument("--lang", type=str, default="english", help="Language of the dataset")
parser.add_argument("--spacy_model", type=str, default="en_core_web_lg", help="Spacy model to use")
parser.add_argument("--save_vectorizers", action="store_true", default=False, help="Save the TFIDF vectorizers")
args = parser.parse_args()

print(f"Arguments: {args}")

main_dir = args.dir

# Get all the files in the directory
file_list = [file for file in listdir(main_dir) if path.isfile(path.join(main_dir, file)) and file.endswith(".json.gz")]

# Set up the variables
if args.mode == "label":
    label_map = {}
else:
    docs_to_process = []
stop = set(stopwords.words(args.lang))
pattern = compile('[^a-zA-Z]+') # Find all non-alphabetic characters 
nlp = spacy.load(args.spacy_model)
nlp.max_length = 6000000 # Needed to avoid out of memory errors

print("Reading files...")
for file in tqdm(file_list):
    with gzip.open(path.join(main_dir, file), "rt", encoding="utf-8") as f:
        data = json.load(f)
        for doc in data:
            sentence = pattern.sub(" ", data[doc]["full_text"])
            sentence = TreebankWordTokenizer().tokenize(sentence.lower()) # Lowercase the sentence and tokenize it, getting an array of words
            temp = [i for i in sentence if i not in stop and len(i) > 1] # Remove stopwords and words with length 1

            if args.mode == "label":
                # If we chose to summarize by label, we concatenate all the texts of the documents with the same label
                for label in data[doc]["eurovoc_classifiers"]:
                    if label not in label_map:
                        label_map[label] = " ".join(temp)
                    else:
                        label_map[label] = " ".join([label_map[label], " ".join(temp)])
            else:
                # If we chose to summarize by document, we just append the text of the document to the list
                docs_to_process.append(" ".join(temp))

del data, doc

print("Tokenizing...")
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

makedirs(path.join(main_dir, args.out_dir), exist_ok=True)

for file in file_list:
    print(f"Summarizing {file}...")
    with gzip.open(path.join(main_dir, file), "rt", encoding="utf-8") as f:
        data = json.load(f)
        for doc in tqdm(list(data.keys())):
            doc_spacy = nlp(data[doc]["full_text"].lower()) # Lowercase the text and tokenize it with Spacy
            text = [sent.text for sent in doc_spacy.sents] # Split the text into sentences
            
            sent_eval = []
            for sentence in text:
                # Tokenize each sentence, remove stopwords and words with length 1
                sentence = pattern.sub(" ", sentence)
                sentence = TreebankWordTokenizer().tokenize(sentence)
                temp = [i for i in sentence if i not in stop and len(i) > 1]
                sent_score = []
                for word in temp:
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
            data[doc]["full_text"] = text
            data[doc]["importance"] = sent_eval
    
    with gzip.open(path.join(main_dir, args.out_dir, file.replace(".json.gz", "_sum_tfidf_l2.json.gz")), "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

if args.save_vectorizers:
    print("Saving vectorizers...")
    with open(path.join(main_dir, args.out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(path.join(main_dir, args.out_dir, "tfidf_vectorizer_vectors.pkl"), "wb") as f:
        pickle.dump(tfidf_vectorizer_vectors, f)

print("Done!")
