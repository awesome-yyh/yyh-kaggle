'''
Author: yangyahe yangyahe@midu.com
Date: 2024-07-21 04:25:49
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2024-07-21 04:25:49
FilePath: /app/yangyahe/kaggel/titanic/explore.py
Description: 
'''
import pandas as pd


train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")

print(train_df.head(10))
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())

women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

Pclass1 = train_df.loc[train_df.Pclass == 1]["Survived"]
print("% of Pclass1 who survived:", sum(Pclass1)/len(Pclass1))

Pclass2 = train_df.loc[train_df.Pclass == 2]["Survived"]
print("% of Pclass2 who survived:", sum(Pclass2)/len(Pclass2))

Pclass3 = train_df.loc[train_df.Pclass == 3]["Survived"]
print("% of Pclass3 who survived:", sum(Pclass3)/len(Pclass3))

baby = train_df.loc[train_df.Age < 10]["Survived"]
print("% of baby who survived:", sum(baby)/len(baby))

# SibSp: 在titanic上的兄弟姐妹及配偶数
sibsp = train_df.loc[train_df.SibSp >= 2]["Survived"]
print("% of sibsp who survived:", sum(sibsp)/len(sibsp))

# Parch: 在titanic上的父母及孩子数
parch = train_df.loc[train_df.Parch >= 2]["Survived"]
print("% of parch who survived:", sum(parch)/len(parch))

# Ticket: 票号

# Fare: 船费
fare = train_df.loc[train_df.Fare >= 31]["Survived"]
print("% of fare who survived:", sum(fare)/len(fare))

# Cabin: 房间号
cabin = train_df.loc[train_df.Cabin.notnull()]["Survived"]
print("% of cabin who survived:", sum(cabin)/len(cabin))

# Embarked: 启航港口	C = Cherbourg, Q = Queenstown, S = Southampton
embarked = train_df.loc[train_df.Embarked == 'S']["Survived"]
print("% of embarked who survived:", sum(embarked)/len(embarked))


# # import torch

# # def tokenize_names(features, labels=None):
# #     features["Name"] = features["Name"].split()
# #     return features, labels

# # # 假设 preprocessed_train_df 是一个包含数据的 DataFrame
# # # 这里只是模拟数据
# # preprocessed_train_df = {'Name': ['John Doe', 'Jane Smith'], 'Survived': [1, 0]}
# # features = {'Name': preprocessed_train_df['Name']}
# # labels = preprocessed_train_df['Survived']

# # features, labels = tokenize_names(features, labels)
# # print(features)

# # def tokenize_names(features, labels=None):
# #     """Divite the names into tokens. TF-train_df can consume text tokens natively."""
# #     features["Name"] =  tf.strings.split(features["Name"])
# #     return features, labels

# # train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df,label="Survived").map(tokenize_names)
# # serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)
