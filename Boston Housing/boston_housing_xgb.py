import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Reference: https://www.cnblogs.com/TimVerion/p/11436001.html

X = load_boston().data
y = load_boston().target

seed = 2020
test_size = 0.2
abnormal_index = [y == 50]
X = X[y != 50]
y = y[y != 50]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))
# print(X.shape) #(506, 13)
# print(X_train.shape)
xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=50,
    # objective='reg:linear'
    # min_child_weight=5,
    # max_delta_step=0,
    # subsample=0.8,
    # colsample_bytree=0.7,
    # reg_alpha=0,
    # reg_lambda=0.4,
    # scale_pos_weight=0.8,
    # silent=True,
    # missing=None,
    # eval_metric='auc',
    # seed=seed,
    # gamma=0
)
print("xgb_model fitting...")
xgb_model.fit(X_train, y_train)

# xgb_model.save_model('boston_housing_xgb.model')

print("xgb_model predicting...")
y_test_pred = xgb_model.predict(X_test)

# predictions = [round(value) for value in y_pred]
# print("predictions: ", predictions)
# print("y_test: ", y_test)

# assert len(y_test_pred) == len(y_test)
# MSE = np.sum(np.power((y_test - y_test_pred), 2)) / len(y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
# print("XGBoost TEST MSE =", MSE)
print("XGBoost TEST mse =", test_mse)

test_number = range(len(y_test))
plt.plot(test_number, y_test, 'r-', lw=2, label='Actual Price')
plt.plot(test_number, y_test_pred, 'g-', lw=4, label='Model Predict')
plt.ylabel('Price')
plt.grid(True)
plt.title('Boston Housing Price Prediction')
plt.show()

# y_pred = xgb_model.predict(X)
# # assert len(y_pred) == len(y)
# # MSE = np.sum(np.power((y - y_pred), 2)) / len(y)
# mse = mean_squared_error(y, y_pred)
# # print("XGBoost ALL MSE =", MSE)
# print("XGBoost ALL mse =", mse)