'''
Author: yangyahe yangyahe@midu.com
Date: 2024-07-21 02:51:50
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2024-07-21 14:29:00
FilePath: /app/yangyahe/kaggel/titanic/sklearn.py
Description: 
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processing import data_processing


# 加载数据
train_df = pd.read_csv("titanic/train.csv")
test_df = pd.read_csv("titanic/test.csv")
print(train_df.head())

X = data_processing(train_df)
y = train_df['Survived']

print(X.head())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
print("=== RandomForestClassifier ===")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 预测并评估
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.6f}')

# Submission
test_X = data_processing(test_df)
X_test_predict = rf_model.predict(test_X)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': X_test_predict})
output.to_csv('titanic/submission.csv', index=False)

print("Submission file has been created.")

print(f"=== rf+GridSearchCV ==")
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='precision')
grid_search.fit(X_train, y_train)

# 最佳参数和模型
best_rf = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')

# Predict on test data
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.6f}')

# Submission
test_X = data_processing(test_df)
X_test_predict = best_rf.predict(test_X)
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': X_test_predict})
output.to_csv('titanic/submission.csv', index=False)

print("Submission file has been created.")
