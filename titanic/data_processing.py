'''
Author: yangyahe yangyahe@midu.com
Date: 2024-07-21 12:50:00
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2024-07-21 14:13:43
FilePath: /app/yangyahe/kaggel/titanic/data_processing.py
Description: 
'''
from sklearn.preprocessing import LabelEncoder

def data_processing(train_df):
    # 数据预处理
    train_df['Age'] = train_df['Age'].fillna(int(train_df['Age'].mean()))
    train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    train_df['Cabin'] = train_df['Cabin'].fillna(train_df['Cabin'].mode()[0])


    le = LabelEncoder()

    train_df['Sex'] = le.fit_transform(train_df['Sex'])
    train_df['Embarked'] = le.fit_transform(train_df['Embarked'])
    train_df['Cabin'] = le.fit_transform(train_df['Cabin'])
    # train_df['Name'] = le.fit_transform(train_df['Name'])
    # train_df['Ticket'] = le.fit_transform(train_df['Ticket'])
    # print(train_df.head())

    # # 分箱化 Age 和 Fare 特征
    # train_df['Age'] = pd.cut(train_df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
    # train_df['Fare'] = pd.cut(train_df['Fare'], bins=[0, 7.91, 14.454, 31, 512], labels=False)

    # 选择特征和标签
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin']  # , 'Cabin'
    X = train_df[features]
    
    return X