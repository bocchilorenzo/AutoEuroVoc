This folder contains various utility scripts with different specific purposes. They can be used when needed, no ordering has to be followed. Following is an explanation on how to use them.

## 00. Convert the model to Local-Sparse-Global attention
This script replaces the full attention in the encoder of the model to Local-Sparse-Global attention. The full explanation can be found in the [paper](https://arxiv.org/abs/2210.15497). In short, it allows the model to handle very long sequences. More information can be found in [this repository](https://github.com/ccdv-ai/convert_checkpoint_to_lsg). The script is convert_lsg.py and it takes the following arguments:
- base_model: the path to the base model
- new_path: the path to the new model
- max_sequence_length: the maximum sequence length. Default is 16384.
- block_size: the block size. Default is 128.
- sparse_block_size: the sparse block size. Default is 128.
- sparsity_factor: the sparsity factor. Default is 2.
- num_global_tokens: the number of global tokens. Default is 7.

## 01. Get the length of the processed datasets with the various seeds
When you forget to write down the length of the processed datasets, you can use this script to get them back. The script is 01-get_dataset_lengths.py and it takes the following arguments:
- data_path: the path to the data folder

## 02. T-test
Useful script that automatically calculates the significance of the difference between scores of two models. It calculates both for one tailed and two tailed tests at 0.1, 0.05, 0.01 and 0.001 alpha levels. The script is 02-t_test.py and it takes the following arguments:
- best: list of comma separated values of the supposed best scores
- worst: list of comma separated values of the other scores

## 03. Update model labels
This script updates the labels of the model. It is particularly useful when you want to change the labels of the model to the ones of the dataset (it should not be necessary if you train the model with the script in this repository). The script is 03-update_model_labels.py and it takes the following arguments:
- model_path: the path to the model
- labels: the path to the labels' JSON file

## 04. Create mappings
Useful if you want to use the inference script with textual labels instead of label ids. The script is 04-create_mappings.py and it takes no arguments. It will create a 'label_mappings' folder in the config directory for each of the label types and it will create the mappings in the various languages available.

## 05. Print the scores of the models
With this script you can directly compare the performance of two different models for the same language. The script will print and also save the t-test for the two models and also the average scores over their seeds in the preferred format, either latex or csv. The script is 05-print_scores.py and it takes the following arguments:
- main_path: the path to the main models
- summ_path: the path to the summarized models
- save_path: the path to save the scores. If not provided, the scores will not be saved
- lang: the language of the models
- t_test: perform a t-test between the main and summarized models
- print_type: latex or csv

## 06. Create graphs for the stats
This script creates graphs for the stats of the models. The script is 06-create_graphs.py and it takes the following arguments:
- stats_path: the path to the folder with the stats files previously generated
- output_path: the path to the folder where to save the graphs

## 07. Get the stats for the data
This script gets the stats for the data regarding the number of labels and documents removed after trimming the dataset with a threshold (using the script in create_dataset/02-remove_few_labels.py). The script is 07-get_dataset_stats.py and it takes the following arguments:
- full: the path to the folder for the full data, organized by language
- truncated: the path with the truncated data, organized by language
- output: the path for the output folder