import numpy as np
import pandas
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# train = pandas.read_csv("data/titanic_train.csv")
test = pandas.read_csv("data/titanic_test.csv")

def preprocess(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    # titanic["child"] = titanic["Age"].apply(lambda x:1 if x < 15 else 0)

    # titanic["Sex"] = titanic["Sex"].apply(lambda x:1 if x == "male" else 0)
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic["Embarked"] = titanic["Embarked"].fillna('S')
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    return titanic

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

# define the test for LinearRegression with KFolder with N_SPLITS
def testRegression(N_SPLITS):
    train_data = pandas.read_csv("data/titanic_train.csv")
    test_data = pandas.read_csv("data/titanic_test.csv")

    train_data = preprocess(train_data)

    alg = LinearRegression()
    # alg = LogisticRegression(random_state=1)
    # scores = cross_validation.cross_val_score(alg, train_data[features], train_data["Survived"], cv=3)

    # cross validation
    kfold = KFold(n_splits = N_SPLITS, shuffle = True, random_state = 1)

    predictions = []
    # print(train_data.shape)(891, 12)
    iteration = 0
    for train, test in kfold.split(train_data):
        # print("test=",test)
        train_predictors = (train_data[features].iloc[train, :])
        train_target = train_data["Survived"].iloc[train]

        alg.fit(train_predictors, train_target)

        test_predictions = alg.predict(train_data[features].iloc[test, :])
        test_label = train_data["Survived"].iloc[test]

        test_predictions[test_predictions > .5] = 1
        test_predictions[test_predictions <= .5] = 0

        # print("test_predictions =",test_predictions[:20])
        # print("test_label =", test_label[:20])
        iteration = iteration + 1

        accuracy = sum(test_predictions == test_label) / len(test_predictions)

        print("KF iter =", iteration, ",accuracy =", accuracy)

# define the test for XGBoost
def testXGBoost():
    train_data = pandas.read_csv("data/titanic_train.csv")
    test_data = pandas.read_csv("data/titanic_test.csv")
    train = preprocess(train_data)
    test = preprocess(test_data)
    clf = XGBClassifier(learning_rate = 0.1, max_depth = 2, silent = True, objective = 'binary:logistic')
    parameters = {
        'n_estimators': [30,32,34,36,38,40,42,44,46,48,50],
        'max_depth':[2,3,4,5,6,7]
    }
    grid_search = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'accuracy', cv = 5)
    grid_search.fit(train[features],train["Survived"])

    print(grid_search.best_params_,grid_search.best_score_)

    predict_data = grid_search.predict(test[features])

# testRegression(5)
testXGBoost()