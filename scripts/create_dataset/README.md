This folder contains various utilities to recreate the datasets to use with the model. Following is an explanation on how to use them. For the arguments of each script, run the script with the -h flag.

## 00. Downloading the data
This is done using the scraper found at https://github.com/bocchilorenzo/scrapelex. Follow the readme there to download the data.

## 01. Extract the documents with eurovoc classifiers
This is the only necessary step that needs to be done in order to use the data with the classifier. To do this, put the downloaded data together in a folder and run the script 01-extract_eurovoc.py. This will create a gzipped json file for each original file, but only containing the documents that were classified and which contain text. The only arguments required are "data_path", which points to the folder containing the downloaded data, and "output_path", which points to the folder where the extracted documents will be saved.

## 02. Remove labels associated with less than a specified amount of documents
Using the extracted documents, it's possible to remove the labels that are associated with less than a specified amount of documents. This is done by running the script 02-remove_few_labels.py. When setting the year range, it's important to use the correct years as the script will base the removal on the bulk of the data, so if the years are wrong it could remove labels that shouldn't be removed.

Any documents that end up having no labels get removed and reported to the user in the console.

## 03. Deduplicate the documents
To deduplicate the documents, run 02-deduplicate_data.py. This script will load the data year by year and remove documents that are identical or almost identical to each other. It uses the library "sentence-transformers" to compute the cosine similarity between the documents, via the "paraphrase_mining" method. The main downside is that the max input length for all the languages aside for English is 128, while for English is 384, so it could remove documents that shouldn't be removed.

The deleted documents are saved in a JSON file (if the argument is used), in the format:

```json
{
    "deleted_id": "similar_id"
}
```

By default, it's False.

## 04. Summarize the documents with a centroid-based summarizer
This summarizer is based on the code at https://github.com/holydrinker/text-summarizer/ and the paper [Centroid-based Text Summarization through Compositionality of Word Embeddings](www.aclweb.org/anthology/W/W17/W17-1003.pdf) by Gaetano Rossiello, Pierpaolo Basile and Giovanni Semeraro.

The code was adapted to allow for the use of either a Word2Vec model or a fastText model. It also has the ability to work with compressed fastText models in order to be usable in an environment with limited resources.

### How to use
NOTE: You can skip step 1 if you already have UDpipe 1 and the models installed or if you want to use the NLTK sentence tokenizer instead of UDpipe's.

1. Install UDpipe 1. You can find installation instructions on https://ufal.mff.cuni.cz/udpipe/1/install. In short, download the release from Github and install the binary (on Windows, copy the folder for either the 32bit or 64bit binary wherever you want and add its path to the PATH environment variable).

2. Download a word embeddings model. We recommend using fastText.

3. Run 04-summarize_dataset.py. Before doing the summarization, the script will check if the UDPipe models are present (only if the udpipe tokenizer is used). If they aren't, they will be downloaded. This script will load the data year by year and summarize the documents using the summarizer downloaded in the previous step. For UDPipe 2, the tokenizer uses the API hosted at http://lindat.mff.cuni.cz/services/udpipe/api/process.

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

## 05. Summarize the documents with a TFIDF-based summarizer
This summarizer works by computing the TFIDF score of the words in the documents. It has two main modes to calculate the values:
- by label: the TFIDF score is calculated for each label.
- by document: the TFIDF score is calculated for each document.
In both cases, the stopwords get removed and the scores to the words are given considering the TFIDF matrix as a whole, without picking the score for the word from the labels or for that specific document. The final score assigned to a sentence depends on the mode chosen:
- mean: for each word, the maximum score is extracted from the TFIDF matrix and then the mean of all the scores in a sentence is calculated.
- max: for each word, the maximum score is extracted from the TFIDF matrix and then the maximum of all the scores in a sentence is calculated.

Before launching the summarization, make sure that the spacy model you want to use is installed on the system, otherwise it will throw an error. If using docker, include the instruction to install the model in the Dockerfile.

The final output follows the same structure as the centroid-based summarizer.