from re import sub

allowedPos = {"PROPN", "VERB", "NOUN", "ADJ", "ADV"}

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
    sentence['text'] = sub(" +", " ", sentence['text']).strip()
    return sentence