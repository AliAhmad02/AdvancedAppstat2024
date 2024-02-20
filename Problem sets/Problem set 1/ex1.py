import pandas as pd
import os
import matplotlib.pyplot as plt

# I know this webscraping is done in sort of a round-about way
# At some point, when I have time, I'll sit down and go through a full
# course on webscraping and HTML/XML parsing. Then I'll probably
# scoff at what I've done below ;). But it'll have to do for now.
def read_data_from_url(url, header, non_num_cols):
    html_read = pd.read_html(url)[0]
    temp_save_path = f"{os.getcwd()}/temp.csv"
    # html_read is a complicated multi-index dataframe
    # Instead of dealing with this, we just export it to a csv
    # and read it in again, which gives something much simpler
    html_read.to_csv(temp_save_path, index=False)
    html_raw = pd.read_csv(temp_save_path, header=header).dropna()
    # No longer need temporary csv file, so it is deleted
    os.remove(temp_save_path)
    # Discard rows where the rank value is not a digit
    mask = html_raw[html_raw.columns[0]].str.isdigit()
    html_cleaned = html_raw[mask].reset_index(drop=True)
    # Transform numerical columns to numerical dtypes
    num_cols = html_cleaned.columns[~html_cleaned.columns.isin(non_num_cols)]
    html_cleaned[num_cols] = html_cleaned[num_cols].apply(pd.to_numeric)
    return html_cleaned


def get_and_plot_hist_conf(
    data, col_name, confs, nbins, data_min, data_max, style_dicts
):
    plt.style.use("seaborn-v0_8-bright")
    plt.figure(figsize=(8, 5))
    for idx, name in enumerate(confs):
        hist_vals = data[col_name][data["Conf"] == name].values
        plt.hist(
            hist_vals,
            bins=nbins,
            label=f"{name}",
            range=(data_min, data_max),
            **style_dicts[idx],
            zorder=idx,
        )
    confs_title = ", ".join(confs)
    plt.title(f"Teams in confs {confs_title}", fontsize=15)
    plt.xlabel("Adjusted Defence", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.legend(fontsize=13, frameon=False)
    plt.show()


if __name__ == "__main__":
    url_2014 = "https://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2024/data/2014KenPomeroy.html"
    # First 17 rows of the csv file contain pure garbage, so they are skipped.
    header = 17
    html_2014_cleaned = read_data_from_url(url_2014, header, ["Team", "Conf", "W-L"])
    adj_conf_2014 = ["ACC", "SEC", "B10", "BSky", "A10"]
    nbins = 7
    adj_conf_2014_min, adj_conf_2014_max = 85, 120
    adj_2014_styledicts = [
        {"histtype": "stepfilled", "alpha": 0.5, "lw": 2, "color": "red"},
        {"histtype": "step", "lw": 4, "color": "yellow"},
        {"histtype": "step", "lw": 3, "color": "blue", "ls": "--"},
        {"histtype": "step", "lw": 2, "color": "black"},
        {"histtype": "step", "lw": 2, "color": "green"},
    ]

    get_and_plot_hist_conf(
        html_2014_cleaned,
        "AdjD",
        adj_conf_2014,
        nbins,
        adj_conf_2014_min,
        adj_conf_2014_max,
        adj_2014_styledicts,
    )
