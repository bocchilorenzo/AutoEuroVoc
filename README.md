# AutoEuroVoc

Collection of Python scripts to create a document multilabel classifier with the corresponding EuroVoc labels. The scripts require at least Python 3.8.

## How it works

It uses a BERT model trained for multilabel classification. It is multilanguage and can be trained on the following languages: `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hu`, `it`, `lt`, `lv`, `mt`, `nl`, `pl`, `pt`, `ro`, `sk`, `sl`, `sv`.

## How to use
First of all, clone the repository:
```bash
git clone https://github.com/bocchilorenzo/AutoEuroVoc.git
```

### Requirements
Before installing the requirements, you will need to install PyTorch. After that, the requirements are listed in the `requirements.txt` file. To install them, run:
```bash
pip install -r requirements.txt
```

### Download the data
The suggested option is to create your own dataset in order to have it always updated. To do so, follow the instructions in the `scripts/create_dataset/README.md` file. Otherwise, you can download it from our [OneDrive folder](https://fbk-my.sharepoint.com/:f:/g/personal/aprosio_fbk_eu/EuC0sZXqi-tEtj24Et25BHYBYkPIjs5eXupNpQ7H_sK1Rw?e=L6PTIn) and place it in the `data` folder inside a folder named with the correct 2-letter language code (`it` for Italian). For now, only the Italian data is available to download, but more languages will be added soon.

### Preprocess the data
To preprocess the data, run the `0-preprocess.py` script. It will create a `data` folder containing the preprocessed data. It has many options, run `python 1-preprocess.py --help` to see them.

### Set the seeds
In the `config/seeds.txt` file, you can set the seeds to use during preprocessing and training. Right now, only one seed is set, but to set more simply write them in separate lines.

### Train the model
To train the model, run the `1-train.py` script. It will create a `models` folder containing the trained model. The model used depends on the language chosen. The list of models used for the various languages is contained in `config/models.yml`. Once again, the script has many options. Run `python 1-train.py --help` to see them.

### Evaluate the model
Once the model has been trained, it can be evaluated by running `2-evaluate.py`. It will print the evaluation metrics on the test set, as well as saving them in the model directory. The script has a few options that can be set, such as batch size and device to use. Run `python 2-evaluate.py --help` to see them.

### Use the model in production
There are two more scripts that can be used after the models were created, namely `3a-inference_single_example.py` and `3b-inference_api_example.py`. The first one is used to make predictions on a single example, while the second one is used to create a basic REST API that can be used to make predictions as a web service. The API is created using FastAPI and can be run by running `uvicorn 3b-inference_api_example:app --host 0.0.0.0 --port 8000`. Check the uvicorn docs for more information. Before running the API, you can add some parameters via environment variables. The parameters are:
- `MODEL_PATH`: the path to the model to use. The default path is `./model`.
- `DEVICE`: the device to use. The default device is `cpu`. To use the GPU, set it to `cuda` followed by the number of the GPU, for example `cuda:0`.
- `PRED_TYPE`: whether to return only the label ids or the label ids and the corresponding labels. The default is `id`, to obtain the labels use `labels`.
- `TOP_K`: the number of labels to return. Do not set it if you don't need it.
- `THRESHOLD`: the threshold to use to filter the labels confidence. Do not set it if you don't need it.

If neither of the last two are set, all the labels will be returned. If `THRESHOLD` is used, `TOP_K` will be ignored.

To send text to the API, make a POST request to `http://localhost:8000/api` with the following body:
```json
{
    "text": "Your text here"
}
```

## TODO

[ ] Add parameters of the scripts in this README

[ ] Show examples of the data

[ ] Show examples of the API