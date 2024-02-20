from ex1 import read_data_from_url, get_and_plot_hist_conf
from ex2 import calc_and_plot_diff_confs, mean_median_diff_not_in_conf

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
confs = ["ACC", "SEC", "B10", "BSky", "A10", "BE"]

adjD_2014_styledicts = [
        {"histtype": "stepfilled", "alpha": 0.5, "lw": 2, "color": "red"},
        {"histtype": "step", "lw": 4, "color": "yellow"},
        {"histtype": "step", "lw": 3, "color": "blue", "ls": "--"},
        {"histtype": "step", "lw": 2, "color": "black"},
        {"histtype": "step", "lw": 2.5, "color": "green"},
        {"histtype": "step", "lw": 1.5, "color": "cyan"},
    ]

adjO_styledicts = [
    {"marker": ".", "color": "red"},
    {"marker": "x", "color": "yellow"},
    {"marker": "d", "color": "blue"},
    {"marker": "+", "color": "black"},
    {"marker": "*", "color": "green"},
    {"marker": "1", "color": "cyan"}
]

nbins = 7
adjD_2014_min, adjD_2014_max = 85, 120
get_and_plot_hist_conf(data_2014, "AdjD", confs, nbins, adjD_2014_min, adjD_2014_max, adjD_2014_styledicts)

calc_and_plot_diff_confs(data_2009, data_2014, confs, "AdjO", adjO_styledicts)

mean_diff_not_in_confs, median_diff_not_in_confs = mean_median_diff_not_in_conf(
    data_2009, data_2014, confs, "AdjO"
)
print(f"\nMean difference, rest of confs: {mean_diff_not_in_confs:.2f}")
print(f"\nMedian difference, rest of confs: {median_diff_not_in_confs:.2f}")
