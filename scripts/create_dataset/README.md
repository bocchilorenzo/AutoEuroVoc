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

## 04. Download the summarizer
The summarizer used is contained in a different repository, namely https://github.com/bocchilorenzo/text-summarizer. Running the script 03-download_summarizer.py will download the summarizer and save it in the folder "text_summarizer". Then, it will create an __init__.py file in order to be able to import it as a module and it will download the udpipe models. Finally, it will also install the requirements for the summarizer. The user still has to do two things, namely:

- download the fastText, compressed fastText or word2vec model to use with the summarizer
- download and install udpipe 1

More instructions on how to do this can be found in the readme of the summarizer repository.

## 05. Summarize the documents
To summarize the documents, run 04-summarize_dataset.py. This script will load the data year by year and summarize the documents using the summarizer downloaded in the previous step. The arguments are:

- data_path: the path to the folder containing the deduplicated documents
- output_path: the path to the folder where the summarized documents will be saved
- summ_lang: the language of the summarizer model. Default is "italian".
- model_path: the path to the summarizer model. Default is "./cc.it.300.bin".
- compressed: whether the model is compressed or not. Default is False.
- tokenizer: the tokenizer to use for the summarizer. It can be "nltk" or "udpipe". Default is "udpipe".
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