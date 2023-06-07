This folder contains various utilities to recreate the datasets to use with the model. Following is an explanation on how to use them.

## 00. Downloading the data
This is done using the scraper found at https://github.com/bocchilorenzo/scrapelex. Follow the readme there to download the data.

## 01. Extract the documents with eurovoc classifiers
This is the only necessary step that needs to be done in order to use the data with the classifier. To do this, put the downloaded data together in a folder and run the script 01-extract_eurovoc.py. This will create a gzipped json file for each original file, but only containing the documents that were classified and which contain text. The only arguments required are "data_path", which points to the folder containing the downloaded data, and "output_path", which points to the folder where the extracted documents will be saved.

## 02. Remove labels associated with less than a specified amount of documents
Using the extracted documents, it's possible to remove the labels that are associated with less than a specified amount of documents. This is done by running the script 02-remove_few_labels.py. The arguments are:

- data_path: the path to the folder containing the extracted documents
- years: the years to consider, comma separated. Default is "all", and it will process all the files it finds in the folder. It's important to use the correct years, as the script will base the removal on the bulk of the data, so if the years are wrong, it could remove labels that shouldn't be removed.
- threshold: the threshold for the number of documents associated with a label. Labels with less documents than this will be removed. Default is 100.

Any documents that end up having no labels get removed and reported to the user in the console.

## 03. Deduplicate the documents
To deduplicate the documents, run 02-deduplicate_data.py. This script will load the data year by year and remove documents that are identical or almost identical to each other. It uses the library "sentence-transformers" to compute the cosine similarity between the documents, via the "paraphrase_mining" method. The main downside is that the max input length for all the languages aside for English is 128, while for English is 384, so it could remove documents that shouldn't be removed. The arguments are:

- data_path: the path to the folder containing the extracted documents
- output_path: the path to the folder where the deduplicated documents will be saved
- threshold: the threshold for the cosine similarity. Documents with a similarity higher than this will be considered duplicates. Default is 0.9.
- batch_size: the batch size for the sentence-transformers model. Default is 6.
- device: the device to use for the sentence-transformers model. Default is "cpu".
- save_deleted: whether to save the deleted documents in a JSON file. The format is:

```json
{
    "deleted_id": "similar_id"
}
```

By default, it's False.

## 04. Summarize the documents
The summarizer is based on the code at https://github.com/holydrinker/text-summarizer/ and the paper [Centroid-based Text Summarization through Compositionality of Word Embeddings](www.aclweb.org/anthology/W/W17/W17-1003.pdf) by Gaetano Rossiello, Pierpaolo Basile and Giovanni Semeraro.

The code was adapted to allow for the use of either a Word2Vec model or a fastText model. It also has the ability to work with compressed fastText models in order to be usable in an environment with limited resources.

### How to use
NOTE: You can skip steps 1, 2 and 3 if you already have UDpipe 1 and the models installed or if you want to use the NLTK sentence tokenizer instead of UDpipe's.

1. Install UDpipe 1. You can find installation instructions on https://ufal.mff.cuni.cz/udpipe/1/install. In short, download the release from Github and install the binary (on Windows, copy the folder for either the 32bit or 64bit binary wherever you want and add its path to the PATH environment variable).

2. Download a word embeddings model. We recommend using fastText.

3. Run 04-summarize_dataset.py. Before doing the summarization, the script will check if the UDPipe models are present. If they aren't, they will be downloaded. This script will load the data year by year and summarize the documents using the summarizer downloaded in the previous step. The arguments are:

- data_path: the path to the folder containing the deduplicated documents
- output_path: the path to the folder where the summarized documents will be saved
- summ_lang: the language of the summarizer model. Default is "italian".
- model_path: the path to the summarizer model. Default is "./cc.it.300.bin".
- compressed: whether the model is compressed or not. Default is False.
- tokenizer: the tokenizer to use for the summarizer. It can be "nltk" or "udpipe". Default is "nltk".
- model_type: the type of the summarizer model. It can be "fasttext" or "word2vec". Default is "fasttext".
- years: the range of years to summarize (extremes included). Default is "2010-2022".

The summarized documents will be formatted in a way that allows the end user to choose how much of the text they want to keep. The format is:

```json
"document_id": {
    "title": "document_title",
    "link": "document_link",
    "eurovoc_classifiers": [
        "classifier_1",
        "classifier_2",
        ...
    ],
    "full_text": [
        "sentence_1",
        "sentence_2",
        ...
    ],
    "importance": [
        0.7326374,
        0.1277499,
        ...
    ]
}
```

This way, the user can either keep the top N most significant sentences or add them up to a specified word length.

### Compatible languages
If you use the NLTK tokenizer, the currently supported languages correspond with those for the PunktSentenceTokenizer, which are: czech, danish, dutch, english, estonian, finnish, french, german, greek, italian, malayalam, norwegian, polish, portuguese, russian, slovene, spanish, swedish, turkish.

For the UDPipe tokenizer, the list is vastly larger and can be found at https://ufal.mff.cuni.cz/udpipe/1/models.

### Where to get the word embedding model
The standard fastText models can be found on https://fasttext.cc/docs/en/crawl-vectors.html, while the compressed fastText models can be found at https://github.com/avidale/compress-fasttext/releases/tag/gensim-4-draft and https://zenodo.org/record/4905385.
