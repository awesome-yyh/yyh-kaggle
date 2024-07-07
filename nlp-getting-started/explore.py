import pandas as pd


filename = "./train.csv"
df_train = pd.read_csv(filename, sep=',', encoding='utf_8_sig', header=0, index_col=None)
print(df_train)

filename = "./test.csv"
df_test = pd.read_csv(filename, sep=',', encoding='utf_8_sig', header=0, index_col=None)
print(df_test)

df_train["text_len"] = df_train["text"].apply(lambda x: len(x))
df_test["text_len"] = df_test["text"].apply(lambda x: len(x))
print(df_train["text_len"].describe())
print(df_test["text_len"].describe())