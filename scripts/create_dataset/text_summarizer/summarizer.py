# coding=utf-8
import numpy as np
from compress_fasttext.models import CompressedFastTextKeyedVectors
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from nltk import download
download("punkt")
download("stopwords")
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import TreebankWordTokenizer
from nltk.data import load
import ufal.udpipe
from yaml import safe_load
from re import sub
from os import path
from copy import deepcopy
from requests import post
from time import sleep
import spacy
spacy.prefer_gpu()


# https://github.com/ufal/udpipe/tree/master/bindings/python/examples
class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(format)
        output = ""
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output

    def write_list(self, sentences):
        """Write given ufal.udpipe.Sentence-s in an iterable list."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat("horizontal")
        output = [output_format.writeSentence(sentence).strip() for sentence in sentences]
        
        return output
    
class UDPipe2:
    def __init__(self, language):
        with open(path.join(path.dirname(path.abspath(__file__)), "models2.yml"), "r") as f:
            model_configs = safe_load(f)
        self.model = model_configs[language]
    
    def tokenize(self, text):
        url = f"http://lindat.mff.cuni.cz/services/udpipe/api/process"
        response = post(url, data={
            "model": self.model,
            "tokenizer": True,
            "output": "horizontal",
            "data": text
        })
        sleep(0.5)
        if response.status_code == 200:
            return response.json()["result"].split("\n")
        else:
            raise Exception("Cannot tokenize text")


class LookupTable:
    def __init__(self, model_path, model_type, compressed=True):
        """
        :param model_path: path to the compressed fasttext model
        :param model_type: type of the model to use (fasttext or word2vec)
        :param compressed: True if the model is compressed, False otherwise
        """
        self.model_type = model_type
        if model_type == "word2vec":
            self.model = KeyedVectors.load_word2vec_format(
                model_path, binary=True, unicode_errors="ignore"
            )
        elif model_type == "fasttext":
            if compressed:
                self.model = CompressedFastTextKeyedVectors.load(
                    path.abspath(model_path)
                )
                self.compressed = True
            else:
                self.model = load_facebook_model(path.abspath(model_path))
                self.compressed = False

    def vec_word(self, word):
        """
        Return the vector of a word if it is in the vocabulary, otherwise return a vector of zeros

        :param word: word to get the vector
        :return: vector of the word
        """
        try:
            if (
                self.model_type == "fasttext" and self.compressed
            ) or self.model_type == "word2vec":
                return self.model[word]
            elif self.model_type == "fasttext" and not self.compressed:
                return self.model.wv[word]
        except KeyError:
            return np.zeros(1)

    def vec_sentence(self, sentence):
        """
        Return the vector of a sentence if it is in the vocabulary, otherwise return a vector of zeros

        :param sentence: sentence to get the vector
        :return: vector of the sentence
        """
        try:
            return self.model.get_sentence_vector(sentence)
        except KeyError:
            return np.zeros(300)

    def unseen(self, word):
        """
        Check if a word is in the vocabulary

        :param word: word to check
        :return: True if the word is not in the vocabulary, False otherwise
        """
        try:
            if (
                self.model_type == "fasttext" and self.compressed
            ) or self.model_type == "word2vec":
                self.model[word]
            elif self.model_type == "fasttext" and not self.compressed:
                return not (word in self.model.wv.key_to_index)
            return False
        except KeyError:
            return True


class Summarizer:
    def __init__(
        self,
        model_path=None,
        model_type="fasttext",
        compressed=True,
        tfidf_threshold=0.3,
        language="italian",
        ngram_range=(1, 1),
        tokenizer="nltk",
        max_length=1000000,
    ):
        """
        :param model_path: path to the compressed fasttext model
        :param model_type: type of the model to use (fasttext or word2vec)
        :param compressed: True if the model is compressed, False otherwise
        :param tfidf_threshold: threshold to filter relevant terms
        :param language: language of the text to summarize
        :param ngram_range: range of ngrams to use
        :param tokenizer: tokenizer to use (udpipe1, udpipe2 or nltk)
        :param max_length: maximum length to pass to spacy's nlp.pipe
        """
        self.lookup_table = LookupTable(model_path, model_type, compressed)
        self.tfidf_threshold = tfidf_threshold
        self.sentence_retriever = []
        self.language = language
        self.ngram_range = ngram_range
        self.model_type = model_type
        self.compressed = compressed
        if tokenizer == "udpipe1":
            self.tokenizer_mode = "udpipe1"
            with open(path.join(path.dirname(path.abspath(__file__)), "models.yml"), "r") as f:
                self.model_configs = safe_load(f)
            self.sent_tokenizer = Model(
                path.join("./models", self.model_configs[language] + ".udpipe")
            )
        elif tokenizer == "udpipe2":
            self.tokenizer_mode = "udpipe2"
            self.sent_tokenizer = UDPipe2(language)
        elif tokenizer == "nltk":
            self.tokenizer_mode = "nltk"
            self.sent_tokenizer = load(f"tokenizers/punkt/{language}.pickle")
        elif tokenizer == "spacy":
            self.tokenizer_mode = "spacy"
            self.nlp = spacy.load("en_core_web_lg")
            self.nlp.max_length = max_length
        else:
            raise ValueError("Invalid tokenizer")

    def _preprocessing(self, text):
        """
        Preprocess the text to summarize

        :param text: text to summarize
        :return: preprocessed text
        """
        # Get splitted sentences
        sentences = self.get_data(text)

        # Store the sentence before process them. We need them to build final summary
        self.sentence_retriever = deepcopy(sentences)

        # Remove punctuation and stopwords
        sentences = self.remove_punctuation_nltk(sentences)
        sentences = self.remove_stopwords(sentences)

        return sentences

    def _gen_centroid(self, sentences):
        """
        Generate the centroid of the document

        :param sentences: sentences of the document
        :return: centroid of the document
        """
        tf = TfidfVectorizer(ngram_range=self.ngram_range)
        tfidf = tf.fit_transform(sentences).toarray().sum(0)
        tfidf = np.divide(tfidf, tfidf.max())
        words = tf.get_feature_names_out()

        relevant_terms = [
            words[i]
            for i in range(len(tfidf))
            if tfidf[i] >= self.tfidf_threshold
            and (
                not self.lookup_table.unseen(words[i])
                if self.model_type == "word2vec"
                else True
            )
        ]

        res = [self.lookup_table.vec_word(term) for term in relevant_terms]
        return sum(res) / len(res)

    def _sentence_vectorizer(self, sentences):
        """
        Vectorize the sentences of the document

        :param sentences: sentences of the document
        :return: dictionary of sentences vectorized
        """
        dic = {}
        for i in range(len(sentences)):
            sum_vec = np.zeros(self.lookup_table.model.vector_size)
            sentence = [
                word
                for word in sentences[i].split(" ")
                if (
                    not self.lookup_table.unseen(word)
                    if self.model_type == "word2vec"
                    else True
                )
            ]

            if sentence:
                for word in sentence:
                    word_vec = self.lookup_table.vec_word(word)
                    sum_vec = np.add(sum_vec, word_vec)
                dic[i] = sum_vec / len(sentence)
        return dic

    def _sentence_selection(self, centroid, sentences_dict):
        """
        Select the sentences of the summary

        :param centroid: centroid of the document
        :param sentences_dict: dictionary of sentences vectorized
        :return: ids+importance of the sentences of the document, sentences of the document
        """
        record = []
        for sentence_id in sentences_dict:
            vector = sentences_dict[sentence_id]
            similarity = 1 - cosine(centroid, vector)
            record.append((sentence_id, vector, similarity))

        full_ids_importance = [(x[0], x[2]) for x in record]

        full_phrases_list = list(
            map(
                lambda sent_id: self.sentence_retriever[sent_id],
                map(lambda t: t[0], full_ids_importance),
            )
        )
        return full_ids_importance, full_phrases_list

    def summarize(self, text):
        """
        Summarize the text

        :param text: text to summarize
        :return: ids+importance of the sentences of the document, sentences of the document
        """
        self._check_params(self.tfidf_threshold)

        # Sentences generation (with preprocessing) + centroid generation
        sentences = self._preprocessing(text)

        centroid = self._gen_centroid(sentences)

        # Sentence vectorization + sentence selection
        sentences_dict = self._sentence_vectorizer(sentences)
        ids_importance, phrases = self._sentence_selection(centroid, sentences_dict)

        return ids_importance, phrases

    def remove_punctuation_nltk(self, data):
        """
        Remove punctuation from the sentences

        :param data: sentences to process
        :return: sentences without punctuation
        """
        return [
            " ".join(TreebankWordTokenizer().tokenize(sentence.lower()))
            for sentence in data
        ]

    def remove_stopwords(self, data):
        """
        Remove stopwords from the sentences

        :param data: sentences to process
        :return: sentences without stopwords
        """
        to_return = []
        try:
            stop = set(stopwords.words(self.language))
        except:
            stop = set(get_stop_words(self.language))
        for sentence in data:
            stopped = ""
            sentence = sentence.lower().split(" ")
            temp = [i for i in sentence if i not in stop]
            for word in temp:
                stopped += word
                stopped += " "
            to_return.append(stopped)
        return to_return

    def get_data(self, text):
        """
        Split the text into sentences

        :param text: text to split
        :return: sentences of the text
        """
        if self.tokenizer_mode != "spacy":
            sentences = self.sent_tokenizer.tokenize(text)
        if self.tokenizer_mode == "udpipe1":
            parsed = self.sent_tokenizer.write_list(sentences)
        elif self.tokenizer_mode == "spacy":
            doc = self.nlp(text)
            parsed = []
            for sent in doc.sents:
                parsed.append(sent.text)
        else:
            parsed = sentences

        fixed_sentences = [self._fix_sentence(s) for s in parsed]
        return fixed_sentences

    @staticmethod
    def _fix_sentence(sentence):
        sentence = sentence.replace("\n", " ")
        sentence = sub(" +", " ", sentence).strip()
        return sentence

    @staticmethod
    def _check_params(tfidf):
        try:
            assert 0 <= tfidf <= 1
        except AssertionError:
            raise ("ERROR: the tfidf threshold is not valid")
