import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold

y = load_boston().target
feature_names = load_boston().feature_names
# print(feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

def dataProcessing(df):
    field_cut = {
        'CRIM' : [0,10,20, 100],
        'ZN' : [-1, 5, 18, 20, 40, 80, 86, 100],
        'INDUS' : [-1, 7, 15, 23, 40],
        'NOX' : [0, 0.51, 0.6, 0.7, 0.8, 1],
        'RM' : [0, 4, 5, 6, 7, 8, 9],
        'AGE' : [0, 60, 80, 100],
        'DIS' : [0, 2, 6, 14],
        'RAD' : [0, 5, 10, 25],
        'TAX' : [0, 200, 400, 500, 800],
        'PTRATIO' : [0, 14, 20, 23],
        'B' : [0, 100, 350, 450],
        'LSTAT' : [0, 5, 10, 20, 40]
    }
    df = df[load_boston().feature_names].copy()
    cut_df = pd.DataFrame()

    for field in field_cut.keys():
        cut_series = pd.cut(df[field], field_cut[field], right=True)
        onehot_df = pd.get_dummies(cut_series, prefix=field)
        cut_df = pd.concat([cut_df, onehot_df], axis=1)
    new_df = pd.concat([df, cut_df], axis=1)
    return new_df

df = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
new_df = dataProcessing(df)

# print(new_df.columns)
# Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
#        'PTRATIO', 'B', 'LSTAT', 'CRIM_(0, 10]', 'CRIM_(10, 20]',
#        'CRIM_(20, 100]', 'ZN_(-1, 5]', 'ZN_(5, 18]', 'ZN_(18, 20]',
#        'ZN_(20, 40]', 'ZN_(40, 80]', 'ZN_(80, 86]', 'ZN_(86, 100]',
#        'INDUS_(-1, 7]', 'INDUS_(7, 15]', 'INDUS_(15, 23]', 'INDUS_(23, 40]',
#        'NOX_(0.0, 0.51]', 'NOX_(0.51, 0.6]', 'NOX_(0.6, 0.7]',
#        'NOX_(0.7, 0.8]', 'NOX_(0.8, 1.0]', 'RM_(0, 4]', 'RM_(4, 5]',
#        'RM_(5, 6]', 'RM_(6, 7]', 'RM_(7, 8]', 'RM_(8, 9]', 'AGE_(0, 60]',
#        'AGE_(60, 80]', 'AGE_(80, 100]', 'DIS_(0, 2]', 'DIS_(2, 6]',
#        'DIS_(6, 14]', 'RAD_(0, 5]', 'RAD_(5, 10]', 'RAD_(10, 25]',
#        'TAX_(0, 200]', 'TAX_(200, 400]', 'TAX_(400, 500]', 'TAX_(500, 800]',
#        'PTRATIO_(0, 14]', 'PTRATIO_(14, 20]', 'PTRATIO_(20, 23]', 'B_(0, 100]',
#        'B_(100, 350]', 'B_(350, 450]', 'LSTAT_(0, 5]', 'LSTAT_(5, 10]',
#        'LSTAT_(10, 20]', 'LSTAT_(20, 40]'],
#       dtype='object')

X = new_df.values
y = load_boston().target
# print(X.shape) # (506, 61)
X = X[y!=50]
y = y[y!=50]
# print(X.shape) # (490, 61)

def random_forest_predict():

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    randomForest_model = RandomForestRegressor(n_estimators=10)

    # kfold = KFold(n_splits = 5, shuffle = True)
    # score_ndarray = cross_val_score(randomForest_model, X, y, cv = kf)
    # print(score_ndarray)
    # print(score_ndarray.mean())
    # [0.87130283 0.83137858 0.83232691 0.83551129 0.86237171]
    # 0.8465782645609599

    randomForest_model.fit(X,y)
    rf_train_predicts = randomForest_model.predict(X)
    # print(all_predicts)

    MSE = np.sum(np.power((y - rf_train_predicts), 2))/len(y)
    print("Random Forest mse =", MSE)

def xgboost_predict():
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import ShuffleSplit

    xgb_model = XGBRegressor(nthread=7)

    cv_split = ShuffleSplit(n_splits=6, train_size=0.7, test_size=0.3)

    # In KFolds, each test set should not overlap, even with shuffle.
    # With KFolds and shuffle, the data is shuffled once at the start, and then divided into the number of desired splits.
    # The test data is always one of the splits, the train data is the rest.
    # In ShuffleSplit, the data is shuffled every time, and then split. This means the test sets may overlap between the splits.

    grid_params = dict(
        max_depth=[3, 4 ,5],
        learning_rate=[0.3, 0.03],
        n_estimators=[100]
    )
    grid = GridSearchCV(xgb_model, grid_params, cv=cv_split, scoring='neg_mean_squared_error')
    grid.fit(X, y)

    print(grid.best_params_)
    # {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 100}

    print('XGBoost rmse =', (-grid.best_score_) ** 0.5)

    xgb_train_predicts = grid.predict(X)
    MSE = np.sum(np.power((y - xgb_train_predicts), 2)) / len(y)
    print("XGBoost mse =", MSE)

# random_forest_predict()
# xgboost_predict()

def lightgbm_predict():
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=8,
        learning_rate=0.05,
        n_estimators=720,
        max_bin=55,
        bagging_fraction=0.8,
        bagging_freq=5,
        feature_fraction=0.2319,
        feature_fraction_seed=9,
        bagging_seed=9,
        min_data_in_leaf=6,
        min_sum_hessian_in_leaf=11
    )
    model_lgb.fit(X, y)
    
    lgb_train_predicts = model_lgb.predict(X)
    MSE = np.sum(np.power((y - lgb_train_predicts), 2)) / len(y)
    print("LightGBM mse =", MSE)

# lightgbm_predict()

def test_shuffle():

    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]])
    y = np.array([0,0,0,0,0,1,1,1,1,1])
    splits = 5

    kfold = KFold(n_splits=splits, shuffle=False, random_state=0)
    skfold = StratifiedKFold(n_splits=splits, shuffle=False, random_state=0)
    # StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同

    shufflesplit = ShuffleSplit(n_splits=splits, test_size=0.2, train_size=0.7, random_state=0)
    stratifiedShufflesplit = StratifiedShuffleSplit(n_splits=splits, random_state=42, test_size=0.3)

    print("KFold:")
    for train_index, test_index in kfold.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)

    print("StratifiedKFold:")
    for train_index, test_index in skfold.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)

    print("ShuffleSplit:")
    for train_index, test_index in shufflesplit.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)

    print("StratifiedShuffleSplit:")
    for train_index, test_index in stratifiedShufflesplit.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)

        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]

test_shuffle()