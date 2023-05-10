import numpy as np
import math
import argparse

# t-values for one-tailed and two-tailed tests, for 0.1, 0.05, 0.01 and 0.001 significance levels
t_values = {
    1: [[3.078, 6.314, 31.821, 318.309], [6.314, 12.706, 63.657, 636.619]],
    2: [[1.886, 2.920, 6.965, 22.327], [2.920, 4.303, 9.925, 31.599]],
    3: [[1.638, 2.353, 4.541, 10.215], [2.353, 3.182, 5.841, 12.924]],
    4: [[1.533, 2.132, 3.747, 7.173], [2.132, 2.776, 4.604, 8.610]],
    5: [[1.476, 2.015, 3.365, 5.893], [2.015, 2.571, 4.032, 6.869]],
    6: [[1.440, 1.943, 3.143, 5.208], [1.943, 2.447, 3.707, 5.959]],
    7: [[1.415, 1.895, 2.998, 4.785], [1.895, 2.365, 3.499, 5.408]],
    8: [[1.397, 1.860, 2.896, 4.501], [1.860, 2.306, 3.355, 5.041]],
    9: [[1.383, 1.833, 2.821, 4.297], [1.833, 2.262, 3.250, 4.781]],
    10: [[1.372, 1.812, 2.764, 4.144], [1.812, 2.228, 3.169, 4.587]],
    11: [[1.363, 1.796, 2.718, 4.025], [1.796, 2.201, 3.106, 4.437]],
    12: [[1.356, 1.782, 2.681, 3.930], [1.782, 2.179, 3.055, 4.318]],
    13: [[1.350, 1.771, 2.650, 3.852], [1.771, 2.160, 3.012, 4.221]],
    14: [[1.345, 1.761, 2.624, 3.787], [1.761, 2.145, 2.977, 4.140]],
    15: [[1.341, 1.753, 2.602, 3.733], [1.753, 2.131, 2.947, 4.073]],
    16: [[1.337, 1.746, 2.583, 3.686], [1.746, 2.120, 2.921, 4.015]],
    17: [[1.333, 1.740, 2.567, 3.646], [1.740, 2.110, 2.898, 3.965]],
    18: [[1.330, 1.734, 2.552, 3.610], [1.734, 2.101, 2.878, 3.922]],
    19: [[1.328, 1.729, 2.539, 3.579], [1.729, 2.093, 2.861, 3.883]],
    20: [[1.325, 1.725, 2.528, 3.552], [1.725, 2.086, 2.845, 3.850]],
    21: [[1.323, 1.721, 2.518, 3.527], [1.721, 2.080, 2.831, 3.819]],
    22: [[1.321, 1.717, 2.508, 3.505], [1.717, 2.074, 2.819, 3.792]],
    23: [[1.319, 1.714, 2.500, 3.485], [1.714, 2.069, 2.807, 3.768]],
    24: [[1.318, 1.711, 2.492, 3.467], [1.711, 2.064, 2.797, 3.745]],
    25: [[1.316, 1.708, 2.485, 3.450], [1.708, 2.060, 2.787, 3.725]],
    26: [[1.315, 1.706, 2.479, 3.435], [1.706, 2.056, 2.779, 3.707]],
    27: [[1.314, 1.703, 2.473, 3.421], [1.703, 2.052, 2.771, 3.690]],
    28: [[1.313, 1.701, 2.467, 3.408], [1.701, 2.048, 2.763, 3.674]],
    29: [[1.311, 1.699, 2.462, 3.396], [1.699, 2.045, 2.756, 3.659]],
    30: [[1.310, 1.697, 2.457, 3.385], [1.697, 2.042, 2.750, 3.646]]
}


def get_diff(best, worst):
    diff = best - worst
    diff_mean = np.mean(diff)

    unbiased_std = math.sqrt(
        np.sum((diff-diff_mean)**2)/(len(best)*(len(best)-1)))

    std_mean_diff = diff_mean/unbiased_std

    return std_mean_diff


def t_test(args):
    best = np.array(args.best.split(",")).astype(float)
    worst = np.array(args.worst.split(",")).astype(float)
    std_mean_diff = get_diff(best, worst)

    max_values = t_values[len(best)-1]

    print(f"\nStandardized mean difference: {std_mean_diff}")

    print("Significance at .1, .05, .01 and .001 levels on the one-tailed test:")
    print(f"{max_values[0][0] < std_mean_diff} {max_values[0][1] < std_mean_diff} {max_values[0][2] < std_mean_diff} {max_values[0][3] < std_mean_diff}")

    print("Significance at .1, .05, .01 and .001 levels on the two-tailed test:")
    print(f"{max_values[1][0] < std_mean_diff} {max_values[1][1] < std_mean_diff} {max_values[1][2] < std_mean_diff} {max_values[1][3] < std_mean_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T-test')
    parser.add_argument('--best', type=str, required=True,
                        help='Comma separated list of supposed best values')
    parser.add_argument('--worst', type=str, required=True,
                        help='Comma separated list of supposed worst values')
    args = parser.parse_args()

    t_test(args)