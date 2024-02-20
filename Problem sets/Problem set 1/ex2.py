from ex1 import read_data_from_url
import matplotlib.pyplot as plt
import numpy as np


def data_joint_teams(data1, data2):
    """Given two dataframes, returns the two dataframes
    where all entries where a team is in one dataframe
    and not the other are removed. Also sorts both dataframes
    by the Team column alphabetically"""
    data1_joint = data1[data1["Team"].isin(data2["Team"].values)].sort_values(
        by=["Team"]
    )
    data2_joint = data2[data2["Team"].isin(data1["Team"].values)].sort_values(
        by=["Team"]
    )
    return data1_joint, data2_joint


def calc_diffs_conf(data1, data2, conf, col_name):
    """Calculate the difference between the values of
    a column for teams in a specific conf (only picks out
    teams that are in the conf in both datasets).
    """
    conf1 = data1[data1["Conf"] == conf]
    conf2 = data2[data2["Conf"] == conf]
    conf1_joint, conf2_joint = data_joint_teams(conf1, conf2)
    diff = conf2_joint[col_name].values - conf1_joint[col_name].values
    return diff, conf1_joint[col_name].values


def calc_and_plot_diff_confs(data1, data2, confs, col_name, styledicts):
    """Loop through a lists of confs, perform calc_diffs_conf and
    visualize the results, including mean and median for each conf."""
    plt.style.use("seaborn-v0_8-bright")
    plt.figure(figsize=(12, 7))
    means_diff_confs = []
    medians_diffs_confs = []
    for idx, name in enumerate(confs):
        diff, conf1_joint_vals = calc_diffs_conf(data1, data2, name, col_name)
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        plt.scatter(
            conf1_joint_vals,
            diff,
            s=50,
            **styledicts[idx],
            label=rf"{name}, $\mu$={mean_diff:.2f}, $\tilde{{x}}={median_diff:.2f}$",
        )
        means_diff_confs.append(mean_diff)
        medians_diffs_confs.append(median_diff)
    plt.xlabel("Adjusted offensive efficiency, 2009", fontsize=17)
    plt.ylabel("Difference in AdjO (2014-2009)", fontsize=17)
    plt.legend(fontsize=13)
    plt.show()

def mean_median_diff_not_in_conf(data1, data2, confs, col_name):
    """Calculate the mean and median of the difference between
    two columns for all teams not in a list of confs."""
    not_in_confs1 = data1[~data1["Conf"].isin(confs)]
    not_in_confs2 = data2[~data2["Conf"].isin(confs)]

    not_in_confs1_joint, not_in_confs2_joint = data_joint_teams(
        not_in_confs1, not_in_confs2
    )

    diff_not_in_confs = (
        not_in_confs2_joint[col_name].values - not_in_confs1_joint[col_name].values
    )
    mean_diff_not_in_confs = np.mean(diff_not_in_confs)
    median_diff_not_in_confs = np.median(diff_not_in_confs)
    return mean_diff_not_in_confs, median_diff_not_in_confs

if __name__ == "__main__":
    url_2014 = "https://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2024/data/2014KenPomeroy.html"
    url_2009 = "https://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2024/data/2009KenPomeroy.html"
    non_num_cols = ["Team", "Conf", "W-L"]
    header = 17

    data_2009 = read_data_from_url(url_2009, header, non_num_cols)
    data_2014 = read_data_from_url(url_2014, header, non_num_cols)

    # Removing the digit at the end of some of the team names
    team_names_digit_end = r"\s+[0-9]+$"
    data_2009["Team"] = data_2009["Team"].str.replace(team_names_digit_end, "", regex=True)
    data_2014["Team"] = data_2014["Team"].str.replace(team_names_digit_end, "", regex=True)
    confs = ["ACC", "SEC", "B10", "BSky", "A10"]

    adjO_styledicts = [
        {"marker": ".", "color": "red"},
        {"marker": "x", "color": "yellow"},
        {"marker": "d", "color": "blue"},
        {"marker": "+", "color": "black"},
        {"marker": "*", "color": "green"},
    ]

    calc_and_plot_diff_confs(data_2009, data_2014, confs, "AdjO", adjO_styledicts)

    mean_diff_not_in_confs, median_diff_not_in_confs = mean_median_diff_not_in_conf(
        data_2009, data_2014, confs, "AdjO"
    )
    print(f"\nMean difference, rest of confs: {mean_diff_not_in_confs:.2f}")
    print(f"\nMedian difference, rest of confs: {median_diff_not_in_confs:.2f}")
