# coding=utf-8
import numpy as np
from compress_fasttext.models import CompressedFastTextKeyedVectors
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from yaml import safe_load
from re import sub
from os import path
from copy import deepcopy
from requests import post
from time import sleep
import spacy
from text_summarizer import Cache
spacy.prefer_gpu()
import json


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
        ngram_range=(1, 1),
        spacy_model="en_core_web_sm",
        spacy_num_threads=8,
        spacy_max_length=1000000,
        cache=None,
        min_sent_length=20,
        max_sent_length=500,
        max_sent_entity_ratio=0.5
    ):
        """
        :param model_path: path to the compressed fasttext model
        :param model_type: type of the model to use (fasttext or word2vec)
        :param compressed: True if the model is compressed, False otherwise
        :param tfidf_threshold: threshold to filter relevant terms
        :param ngram_range: range of ngrams to use
        :param spacy_max_length: maximum length to pass to spacy's nlp.pipe
        """
        self.lookup_table = LookupTable(model_path, model_type, compressed)
        self.tfidf_threshold = tfidf_threshold
        self.sentence_retriever = []
        self.ngram_range = ngram_range
        self.model_type = model_type
        self.compressed = compressed
        self.spacy_num_threads = spacy_num_threads

        self.min_sent_length=min_sent_length
        self.max_sent_length=max_sent_length
        self.max_sent_entity_ratio=max_sent_entity_ratio

        self.cache = None
        if cache:
            self.cache = Cache(cache)

        self.tokenizer_mode = "spacy"

        print(f"Loading model for spaCy:", spacy_model)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print('Downloading language model for spaCy')
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

        self.nlp.max_length = spacy_max_length

    def _preprocessing(self, text, docID):
        """
        Preprocess the text to summarize

        :param text: text to summarize
        :return: preprocessed text
        """
        # Get splitted sentences
        sentences = []
        if self.cache:
            s = self.cache.getFile(docID)
            if s:
                sentences = json.loads(s)
            else:
                sentences = self.get_data(text)
                self.cache.writeFile(docID, json.dumps(sentences))
        else:
            sentences = self.get_data(text)

        # Store the sentence before process them. We need them to build final summary
        self.sentence_retriever = [sentence['text'] for sentence in sentences]

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
            c = cosine(centroid, vector)
            if c == 0:
                similarity = 0
            else:
                similarity = 1 - c
            record.append((sentence_id, vector, similarity))

        full_ids_importance = [(x[0], x[2]) for x in record]

        full_phrases_list = list(
            map(
                lambda sent_id: self.sentence_retriever[sent_id],
                map(lambda t: t[0], full_ids_importance),
            )
        )
        return full_ids_importance, full_phrases_list

    def summarize(self, text, docID=None):
        """
        Summarize the text

        :param text: text to summarize
        :return: ids+importance of the sentences of the document, sentences of the document
        """
        self._check_params(self.tfidf_threshold)

        # Sentences generation (with preprocessing) + centroid generation
        sentences = self._preprocessing(text, docID)

        # each item of sentences is a complex object, let's simplify it
        simplifiedSentences = []
        allowedPos = {"PROPN", "VERB", "NOUN", "ADJ", "ADV"}

        empty = True
        for sentence in sentences:
            simplifiedSentence = []
            l = len(sentence['text'])

            if l < self.min_sent_length:
                pass
            elif l > self.max_sent_length and self.max_sent_length != 0:
                pass
            else:
                nerCount = 0
                for i in range(len(sentence["token"])):
                    if sentence["ner"][i]:
                        nerCount += 1
                    pos = sentence["pos"][i]
                    if pos in allowedPos:
                        simplifiedSentence.append(sentence["lemma"][i])
                if nerCount / len(sentence['token']) > self.max_sent_entity_ratio:
                    simplifiedSentence = []

            # import pdb;pdb.set_trace()
            empty = empty and len(simplifiedSentence) == 0
            simplifiedSentences.append(" ".join(simplifiedSentence))

        if empty:
            simplifiedSentences = []
            for sentence in sentences:
                simplifiedSentence = []
                print(sentence)
                for i in range(len(sentence["token"])):
                    pos = sentence["pos"][i]
                    if pos in allowedPos:
                        simplifiedSentence.append(sentence["lemma"][i])
                simplifiedSentences.append(" ".join(simplifiedSentence))

        try:
            centroid = self._gen_centroid(simplifiedSentences)
        except:
            return "", ""

        # Sentence vectorization + sentence selection
        sentences_dict = self._sentence_vectorizer(simplifiedSentences)
        ids_importance, phrases = self._sentence_selection(centroid, sentences_dict)

        return ids_importance, phrases

    def get_data(self, text):
        """
        Split the text into sentences

        :param text: text to split
        :return: sentences of the text
        """

        parsed = []
        parts = text.split("\n")
        parts = [x.strip() for x in parts]
        if self.spacy_num_threads == 1:
            for p in parts:
                doc = self.nlp(p)
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
            docs = self.nlp.pipe(parts, n_process=self.spacy_num_threads)
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

        fixed_sentences = [self._fix_sentence(s) for s in parsed]
        return fixed_sentences

    @staticmethod
    def _fix_sentence(sentence):
        sentence['text'] = sentence['text'].replace("\n", " ")
        sentence['text'] = sub(" +", " ", sentence['text']).strip()
        return sentence

    @staticmethod
    def _check_params(tfidf):
        try:
            assert 0 <= tfidf <= 1
        except AssertionError:
            raise ("ERROR: the tfidf threshold is not valid")

