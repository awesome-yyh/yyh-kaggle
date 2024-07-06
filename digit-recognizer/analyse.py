import pandas as pd


filename = "./train.csv"
train_csv = pd.read_csv(filename, skiprows=0, sep=',', encoding='utf_8_sig', header=0, index_col=None)
print(train_csv)