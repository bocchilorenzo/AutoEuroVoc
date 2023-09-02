import numpy as np
import scipy.stats as stats
from os import path, listdir
import json
import argparse

def get_scores(main_path, summarized_path, save_path, lang="en", t_test=False, print_type="latex"):
    def _replace_keys(array):
        scores_dict, max_scores_dict = {}, {}
        for score in array:
            for key in score:
                new_key = key
                if 'ndcg' in key:
                    new_key = key.replace("ndcg_", "NDCG@")
                elif 'roc_auc' in key:
                    new_key = key.replace("roc_auc", "ROC-AUC")
                
                if new_key not in scores_dict:
                    scores_dict[new_key] = []
                scores_dict[new_key].append(score[key])
        
        for key in scores_dict:
            scores_dict[key] = np.array(scores_dict[key])
            max_scores_dict[key] = scores_dict[key].max()
            scores_dict[key] = scores_dict[key].mean()
        
        return scores_dict, max_scores_dict
    
    def _t_test(best, worst):
        best = np.array(best).astype(float)
        worst = np.array(worst).astype(float)

        t_result_one = stats.ttest_ind(best, worst, equal_var=True, alternative="greater")
        t_result_two = stats.ttest_ind(best, worst, equal_var=True, alternative="two-sided")

        to_return = ""
        to_return += "\nOne-tailed result:\n"
        to_return += f"t-statistic: {t_result_one.statistic}\n"
        to_return += f"p-value: {t_result_one.pvalue}\n"
        to_return += f"Significance at .1, .05, .01 and .001 levels: {t_result_one.pvalue < .1} {t_result_one.pvalue < .05} {t_result_one.pvalue < .01} {t_result_one.pvalue < .001}\n"

        to_return += "\nTwo-tailed result:\n"
        to_return += f"t-statistic: {t_result_two.statistic}\n"
        to_return += f"p-value: {t_result_two.pvalue}\n"
        to_return += f"Significance at .1, .05, .01 and .001 levels: {t_result_two.pvalue < .1} {t_result_two.pvalue < .05} {t_result_two.pvalue < .01} {t_result_two.pvalue < .001}\n"

        return t_result_one.pvalue, t_result_two.pvalue, to_return
    
    data = []
    datasumm = []
    data_seeds = []
    datasumm_seeds = []

    for data_path in [main_path, summarized_path if summarized_path else main_path]:
        for model_folder in sorted(listdir(path.join(data_path, lang))):
            data_seeds.append(model_folder) if data_path == main_path else datasumm_seeds.append(model_folder)
            checkpoint = [folder for folder in listdir(path.join(data_path, lang, model_folder)) if "checkpoint" in folder][0]

            with open(path.join(data_path, lang, model_folder, checkpoint, "evaluation", "metrics.json"), "r") as f:
                data.append(json.load(f)) if data_path == main_path else datasumm.append(json.load(f))
    
    assert len(data) == len(datasumm), "The number of models in the main path and the summarized path must be the same"
    assert all([data_seeds[i] == datasumm_seeds[i] for i in range(len(data_seeds))]), "The seeds of the models in the main path and the summarized path must be the same"

    divider = " & " if print_type == "latex" else ","
    end = " \\\\ \\hline\n" if print_type == "latex" else "\n"
    file_extension = "txt" if print_type == "latex" else "csv"

    if t_test:
        if not summarized_path:
            raise ValueError("You must provide a path for the summarized models to perform a t-test")
        
        print("T-test per metric")
        significance_scores = {}
        to_write = ""
        for key in data[0].keys():
            std_w = []
            std_b = []
            for i in range(len(data)):
                std_w.append(str(data[i][key]))
                std_b.append(str(datasumm[i][key]))
            to_write += "\n" + key
            one_tailed, two_tailed, text = _t_test(std_b, std_w)
            significance_scores[key] = {
                "scores_w": std_w,
                "scores_b": std_b,
                "one_tailed": one_tailed,
                "two_tailed": two_tailed,
            }
            to_write += text
        print(to_write)
        
        if save_path != "":
            with open(path.join(save_path, f"t_test_detailed.txt"), "w") as f:
                f.write(to_write)

    print("\n\nT-test per metric and model type")
    to_write = f"Metric{divider}TC{divider}MT{divider}DO{end}"
    for key in significance_scores:
        if "_mt" in key or "_do" in key:
            continue
        to_write += (
            f"{key.split('_')[0].capitalize()}{' ' + key.split('_')[1].capitalize() if len(key.split('_')) > 1 else ''}{divider}"
            f"{significance_scores[key]['one_tailed'] < .05}{divider}{significance_scores[key + '_mt']['one_tailed'] < .05}{divider}"
            f"{significance_scores[key + '_do']['one_tailed'] < .05}"
        )
        to_write += end
    print(to_write)

    if save_path != "":
        with open(path.join(save_path, f"t_test.{file_extension}"), "w") as f:
            f.write(to_write)    

    scores_dict_base, _ = _replace_keys(data)
    scores_dict_summ, _ = _replace_keys(datasumm)

    to_write = f"{divider}Normal{divider*3}Summarized{divider*2}{end}"
    to_write += f"{divider}TC{divider}MT{divider}DO"
    if summarized_path:
        to_write += f"{divider}TC{divider}MT{divider}DO"
    to_write += end
    
    for i in range(len(scores_dict_base)):
        name = list(scores_dict_base)[i]
        if "_mt" in name or "_do" in name:
            continue
        to_write += (
            f"{name.split('_')[0].capitalize()}{' ' + name.split('_')[1].capitalize() if len(name.split('_')) > 1 else ''}"
            f"{divider}{list(scores_dict_base.values())[i]*100:.2f}{divider}{scores_dict_base[list(scores_dict_base.keys())[i] + '_mt']*100:.2f}"
            f"{divider}{scores_dict_base[list(scores_dict_base.keys())[i] + '_do']*100:.2f}"
        )
        if summarized_path:
            to_write += (
                f"{divider}{list(scores_dict_summ.values())[i]*100:.2f}{divider}{scores_dict_summ[list(scores_dict_summ.keys())[i] + '_mt']*100:.2f}"
                f"{divider}{scores_dict_summ[list(scores_dict_summ.keys())[i] + '_do']*100:.2f}"
            )
        to_write += end
    print(to_write)
    
    if save_path != "":
        with open(path.join(save_path, f"scores.{file_extension}"), "w") as f:
            f.write(to_write)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print the scores of the models in the given paths', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--main_path', type=str, required=True,
                        help='Path to the main models.')
    parser.add_argument('--summ_path', type=str, required=False,
                        help='Path to the summarized models.')
    parser.add_argument('--save_path', type=str, required=False, default="",
                        help='Path to save the scores. If not provided, the scores will not be saved.')
    parser.add_argument('--lang', type=str, required=False, default="en",
                        help='Language of the models.')
    parser.add_argument('--t_test', action='store_true',
                        help='Perform a t-test between the main and summarized models.')
    parser.add_argument('--print_type', type=str, required=False, default="latex",
                        choices=["latex", "csv"], help='Print type: latex or csv.')
    args = parser.parse_args()

    get_scores(args.main_path, args.summ_path, args.save_path, args.lang, args.t_test, args.print_type)