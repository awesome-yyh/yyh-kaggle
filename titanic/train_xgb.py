'''
Author: yangyahe yangyahe@midu.com
Date: 2024-07-21 02:51:50
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2024-07-21 14:25:06
FilePath: /app/yangyahe/kaggel/titanic/sklearn.py
Description: 
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processing import data_processing


# 加载数据
train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")
print(train_df.head())

X = data_processing(train_df).drop(columns=['Cabin'])
y = train_df['Survived']

print(X.head())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
print("==== xgboost ====")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'random_state': 42,
    'learning_rate': 0.2,
    'tree_method': 'hist',  # 使用 GPU 加速
    'predictor': 'gpu_predictor',
    'device': "cuda:0"
}
# 训练模型
evallist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params, dtrain, num_boost_round=50, evals=evallist, early_stopping_rounds=10)

# 预测
y_pred = xgb_model.predict(dtest)
y_pred = (y_pred > 0.5).astype(int)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost Accuracy with GPU: {accuracy:.6f}')

# Submission
test_X = data_processing(test_df).drop(columns=['Cabin'])
print(test_X)
dtest = xgb.DMatrix(test_X)
y_pred = xgb_model.predict(dtest)
X_test_predict = (y_pred > 0.5).astype(int)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': X_test_predict})
output.to_csv('titanic/submission.csv', index=False)

print("xgboost Submission file has been created.")

# 保存模型
import joblib
joblib.dump(xgb_model, 'xgboost_model.pkl')

# 加载模型
loaded_model = joblib.load('xgboost_model.pkl')

new_data = [[3, 1, 2, 1, 0, 1, 1],
            [1, 0, 3, 0, 0, 3, 0]]

feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
new_data_df = pd.DataFrame(new_data, columns=feature_columns)

# 转换为 DMatrix 格式
dnew = xgb.DMatrix(new_data_df)  

# 使用加载的模型进行预测
new_predictions = loaded_model.predict(dnew)
new_predictions = (new_predictions > 0.5).astype(int)
print(f'新数据的预测结果: {new_predictions}')


print(f"=== xgb+GridSearchCV ==")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'tree_method': ['hist'],  # 使用 GPU 加速
    'device': ["cuda:0"]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 最佳参数和模型
best_xgb = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')

# Make predictions
y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.6f}')
