from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


import numpy as np
import pandas as pd

from sklearn.svm import SVC
import xgboost as xgb

from matplotlib import pyplot


class modeller():
    def __init__(self, data_frame, response_column):
        self.input_df = data_frame
        self.response_column = response_column
        self.X = data_frame.drop([response_column], 1)
        self.y = data_frame[[response_column]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2)
        print("\nNumber of rows read: " + str(df.shape[0]))
        print("Number of colums read: " + str(df.shape[1]))
        print("\nTrain X Rows: " + str(self.X_train.shape[0]))
        print("Train y Rows: " + str(self.y_train.shape[0]))
        print("Train X Colums:"+str(self.X_train.shape[1]))
        print("Train y Colums:"+str(self.y_train.shape[1]))

    def _xgb(self):
        model = xgb.XGBClassifier()

        # ParameterTuning
        max_depth = range(3, 20, 5)
        min_child_weight = range(1, 6, 2)
        #gamma = range(0, 10, 1)
        learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        n_estimators = range(50, 100, 50)
        param_grid = dict(learning_rate=learning_rate,
                          max_depth=max_depth,
                          min_child_weight=min_child_weight,
                          #gamma=gamma,
                          n_estimators=n_estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        grid_search = GridSearchCV(
            model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(
            self.X_train, self.y_train.values.ravel())

        # Model Building using best params
        print(grid_result.best_params_)
        e_learning_rate = grid_result.best_params_['learning_rate']
        e_max_depth = grid_result.best_params_['max_depth']
        e_min_child_weight = grid_result.best_params_['min_child_weight']
        #e_gamma = grid_result.best_params_['gamma']
        e_n_estimators = grid_result.best_params_['n_estimators']
        e_model = xgb.XGBClassifier(learning_rate=e_learning_rate,
                                    max_depth=e_max_depth,
                                    min_child_weight=e_min_child_weight,
                                    #gamma=e_gamma,
                                    n_estimators=e_n_estimators,
                                    cv=kfold)
        e_model.fit(self.X_train, self.y_train.values.ravel())

        # Accuracy testing
        xgb_pred_class = e_model.predict(self.X_test)
        predictions = [round(value) for value in xgb_pred_class]

        print("XGB % acc =" +
              str(accuracy_score(self.y_test.values.ravel(), predictions)*100)+"%")
        xgb.plot_importance(e_model)
        # pyplot.show()

    def _rf(self):
        clf = RandomForestClassifier(n_jobs=4, random_state=0)
        clf.fit(self.X_train, self.y_train.values.ravel())
        rf_pred_class = clf.predict(self.X_test)
        predictions = [round(value) for value in rf_pred_class]

        print("RF % acc =" +
              str(accuracy_score(self.y_test.values.ravel(), predictions)*100)+"%")

    def _svm(self):
        model = svm.SVC(kernel='rbf', C=1, gamma=1)
        model.fit(self.X_train, self.y_train.values.ravel())
        predictions = model.predict(self.X_test)
        print("SVM % acc =" +
              str(accuracy_score(self.y_test.values.ravel(), predictions)*100)+"%")


df = pd.read_csv('z6.csv')
md = modeller(df, "major")

# df = pd.read_csv('diabetes.csv')
# md = modeller(df, "Outcome")
md._xgb()
# md._rf()
