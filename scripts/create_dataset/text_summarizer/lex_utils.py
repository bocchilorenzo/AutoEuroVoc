import re
from pagerange import PageRange
import os

allowedPos = {"PROPN", "VERB", "NOUN", "ADJ", "ADV"}
reYearFile = r"^([0-9]{4})([^0-9].*)?\.json(\.gz)?$"

def add_argument(parser, arg_label, required=None):
    if arg_label == "years":
        required = False if required is None else required
        parser.add_argument("--years", metavar="YEARS", type=str, required=required, default="all", help="Years to consider. If not specified, all the years will be considered. Multiple years can be specified either as a comma-separated list (e.g. 2019,2020,2021) or as a range (e.g. 2019-2021).")
    if arg_label == "data_path":
        required = True if required is None else required
        parser.add_argument("--data_path", metavar="DIR", type=str, help="Path to the folder containing the .json.gz files", required=required)
    if arg_label == "output_path":
        required = True if required is None else required
        parser.add_argument("--output_path", metavar="DIR", type=str, help="Path to the folder where the output files will be saved", required=required)
    if arg_label == "seed_file":
        required = True if required is None else required
        parser.add_argument("--seed_file", metavar="FILE", type=str, help="Path to the JSON file with seeds information", required=required)
    if arg_label == "seeds":
        parser.add_argument("--seeds", type=str, default="110,221,332", help="Seeds to be used for the randomization and creating the data splits, separated by a comma (e.g. 110,221).")

def init_procedure(input_path, output_path, years=None, verbose=True):
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    if years:
        years = get_years(years, input_path, verbose)
    return years

def get_years(years, directory, verbose=True):

    outYears = {}
    for yearFile in os.listdir(directory):
        m = re.match(reYearFile, yearFile)
        if os.path.isfile(os.path.join(directory, yearFile)) and m:
            outYears[m.group(1)] = yearFile

    if years and years != "all":
        years = PageRange(years).pages
        years = set([str(year) for year in years])
        outYears = {k: outYears[k] for k in years}

    if verbose:
        print(f"### Years to process: {', '.join(outYears.keys())}")

    return outYears

def get_data(text, nlp, spacy_num_threads):
    """
    Split the text into sentences

    :param text: text to split
    :return: sentences of the text
    """

    parsed = []
    parts = text.split("\n")
    parts = [x.strip() for x in parts]
    if spacy_num_threads == 1:
        for p in parts:
            doc = nlp(p)
            for sent in doc.sents:
                thisSentence = {}
                thisSentence['text'] = sent.text.strip()
                if not thisSentence['text']:
                    continue
                thisSentence['token'] = []
                thisSentence['lemma'] = []
                thisSentence['pos'] = []
                thisSentence['ner'] = []
                for token in sent:
                    thisSentence['token'].append(token.text)
                    thisSentence['lemma'].append(token.lemma_)
                    thisSentence['pos'].append(token.pos_)
                    thisSentence['ner'].append(token.ent_type_)
                parsed.append(thisSentence)
    else:
        docs = nlp.pipe(parts, n_process=spacy_num_threads)
        for doc in docs:
            for sent in doc.sents:
                thisSentence = {}
                thisSentence['text'] = sent.text.strip()
                if not thisSentence['text']:
                    continue
                thisSentence['token'] = []
                thisSentence['lemma'] = []
                thisSentence['pos'] = []
                thisSentence['ner'] = []
                for token in sent:
                    thisSentence['token'].append(token.text)
                    thisSentence['lemma'].append(token.lemma_)
                    thisSentence['pos'].append(token.pos_)
                    thisSentence['ner'].append(token.ent_type_)
                parsed.append(thisSentence)

    fixed_sentences = [_fix_sentence(s) for s in parsed]
    return fixed_sentences

def _fix_sentence(sentence):
    sentence['text'] = sentence['text'].replace("\n", " ")
    sentence['text'] = re.sub(" +", " ", sentence['text']).strip()
    return sentence