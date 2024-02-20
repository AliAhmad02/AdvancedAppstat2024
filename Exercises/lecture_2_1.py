import pandas as pd

data = pd.read_csv("Advanced Appstat/Exercises/Data/FranksNumbers.txt", delimiter="\t", names=["X", "Y"])
data = data.drop(0)
idx_rows = data[data["X"].str.contains("Data set")].index.to_list()
idx_rows.append(data.index[-1] + 1)
data_list = []
for i in range(len(idx_rows) - 1):
    data_set = data.loc[idx_rows[i]+1:idx_rows[i+1]-1].reset_index(drop=True)
    data_set = data_set.apply(pd.to_numeric)
    data_list.append(data_set)

X_means = [df["X"].mean() for df in data_list]
X_vars = [df["X"].var() for df in data_list]

Y_means = [df["Y"].mean() for df in data_list]
Y_vars = [df["Y"].var() for df in data_list]
