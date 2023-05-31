import numpy as np
import scipy.stats as stats
import argparse

def t_test(args):
    best = np.array(args.best.split(",")).astype(float)
    worst = np.array(args.worst.split(",")).astype(float)

    t_result_one = stats.ttest_ind(best, worst, equal_var=True, alternative="greater")
    t_result_two = stats.ttest_ind(best, worst, equal_var=True, alternative="two-sided")

    print(f"\nOne-tailed result:")
    print(f"t-statistic: {t_result_one.statistic}")
    print(f"p-value: {t_result_one.pvalue}")
    print(f"Significance at .1, .05, .01 and .001 levels: {t_result_one.pvalue < .1} {t_result_one.pvalue < .05} {t_result_one.pvalue < .01} {t_result_one.pvalue < .001}")

    print(f"\nTwo-tailed result:")
    print(f"t-statistic: {t_result_two.statistic}")
    print(f"p-value: {t_result_two.pvalue}")
    print(f"Significance at .1, .05, .01 and .001 levels: {t_result_two.pvalue < .1} {t_result_two.pvalue < .05} {t_result_two.pvalue < .01} {t_result_two.pvalue < .001}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T-test of the results from the training. This is a test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances.')
    parser.add_argument('--best', type=str, required=True,
                        help='Comma separated list of supposed best values')
    parser.add_argument('--worst', type=str, required=True,
                        help='Comma separated list of supposed worst values')
    args = parser.parse_args()

    t_test(args)