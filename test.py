import pandas as pd 
import numpy as np 
import sys 

from datetime import datetime

import optuna

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin

import matplotlib.pyplot as plt
from argparse import Namespace





class Model():
    def __init__(self, train_input, train_labels, test_input,  test_labels, hyperparameters=None):
        ''' 
        Initialize the Model object with hyperparameters.
        @param args: Command line arguments or other configurations
        @param hyperparameters: Dictionary of model hyperparameters
        '''
        self.train_input = train_input
        self.train_labels = train_labels
        self.test_input = test_input
        self.test_labels = test_labels

        # Configure which features are numerical and which are not
        num_features = ['amount_tsh', 'gps_height', 'longitude','latitude', 'population', 'construction_year','years_since_construction']
        cat_features = [feature for feature in self.train_input.columns if feature not in num_features]

        encoder_preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),  # Impute numerical features
                            ('scaler', StandardScaler())
                        ]), num_features),
                        ('cat', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical features
                            ('encoder', TargetEncoder())
                        ]), cat_features)
                    ]
                )

        self.hyperparameters = hyperparameters or {}

        self.model = Pipeline(steps=[('preprocessor', encoder_preprocessor),
                            ('classifier', RandomForestClassifier(
                                    n_estimators=self.hyperparameters.get('n_estimators', 100),
                                    max_depth=self.hyperparameters.get('max_depth', None),
                                    min_samples_split=self.hyperparameters.get('min_samples_split', 2),
                                    min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
                                    max_features=self.hyperparameters.get('max_features', 'sqrt')
                            ))])

    def train(self):
        ''' Train the model on the provided dataset '''
        self.model.fit(self.train_input, self.train_labels)

    def predict(self):
        ''' Predict using the trained model '''
        return self.model.predict(self.test_input)
    

def get_top_k_frequent(df, column_name, k=100):
        top_k_frequent = df[column_name].value_counts().head(k).index.tolist()
        return top_k_frequent
    
def replace_col_with_topk(df, column_name, top_k):
    # Identify values in `column_name` that are not in `top_k`
    condition = ~df[column_name].isin(top_k)
    
    # Replace values in `column_name` based on the condition
    df.loc[condition, column_name] = 'other'

def preprocess_data(df):
    '''
        Pre-processed input data - performs feature engineering 
        @param (df) : data frame representing inputs 
    '''

    # Modify date_recorded column to days from now since days recorded
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    year = df['date_recorded'].dt.year 
    df['years_since_construction'] = (year - df['construction_year'])

    drop_columns = ['date_recorded', 'num_private', 'id', 'wpt_name', 'subvillage']
    df.drop(drop_columns, axis=1, inplace=True)

    reduce_columns = ['ward','funder','installer', 'scheme_name']

    for column in reduce_columns:
        top_k = get_top_k_frequent(df, column)
        replace_col_with_topk(df, column, top_k)

    return df


def objective(trial):
    # model_type = trial.suggest_categorical('model_type', ['RandomForestClassifier', 'GradientBoosterClassifier', 'MLPClassifier'])
    
    model_type = 'HistGradientBoosterClassifier'

    hyperparameters = {}

    if model_type == 'RandomForestClassifier':
        hyperparameters = {
            "n_estimators" : trial.suggest_int('n_estimators', 10, 500),
            "max_depth" : trial.suggest_int('max_depth', 2, 32),
            "min_samples_split" : trial.suggest_int('min_samples_split', 2, 20),
            "min_samples_leaf" : trial.suggest_int('min_samples_leaf', 1, 20),
            "max_features" : trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        
    elif model_type == 'GradientBoosterClassifier':
        hyperparameters = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 10, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
    
    elif model_type == 'MLPClassifier':
        hyperparameters = {
            "hidden_layer_sizes": eval(trial.suggest_categorical('MLPClassifier_hidden_layer_sizes', ['(50,)', '(100,)', '(50, 50)', '(100, 50)'])),
            "activation": trial.suggest_categorical('MLPClassifier_activation', ['identity', 'logistic', 'tanh', 'relu']),
            "solver": trial.suggest_categorical('MLPClassifier_solver', ['sgd', 'adam']),
            "alpha": trial.suggest_float('MLPClassifier_alpha', 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float('MLPClassifier_learning_rate_init', 1e-5, 1e-1, log=True),
            "max_iter": trial.suggest_int('MLPClassifier_max_iter', 200, 1000),
            "batch_size": trial.suggest_categorical('MLPClassifier_batch_size', [64, 128, 256, 'auto']),
        }

        if hyperparameters["solver"] == 'sgd':
            hyperparameters["learning_rate"] = trial.suggest_categorical('MLPClassifier_learning_rate', ['constant', 'invscaling', 'adaptive'])
    elif model_type == 'LogisticRegressionClassifier':
        hyperparameters = {
            "learning_rate" : trial.suggest_float('learning_rate', 0.01, 1.0),
            "max_iter" : trial.suggest_int('max_iter', 10, 1000),
            "max_depth" : trial.suggest_int('max_depth', 1, 32),
            "min_samples_leaf" : trial.suggest_int('min_samples_leaf', 5, 100),
            "l2_regularization" : trial.suggest_float('l2_regularization', 0.0, 1.0),
            "max_bins" : trial.suggest_int('max_bins', 50, 255),
            "min_samples_to_split" : trial.suggest_int('min_samples_to_split', 2, 100)
        }
    elif model_type == "HistGradientBoosterClassifier":
        hyperparameters = {
            "learning_rate": trial.suggest_float('HistGradientBoostingClassifier_learning_rate', 0.01, 1.0),
            "max_iter": trial.suggest_int('HistGradientBoostingClassifier_max_iter', 10, 1000),
            "max_depth": trial.suggest_int('HistGradientBoostingClassifier_max_depth', 1, 32),
            "min_samples_leaf": trial.suggest_int('HistGradientBoostingClassifier_min_samples_leaf', 5, 100),
            "l2_regularization": trial.suggest_float('HistGradientBoostingClassifier_l2_regularization', 0.0, 1.0),
            "max_bins": trial.suggest_int('HistGradientBoostingClassifier_max_bins', 50, 255),
        }

    input = pd.read_csv('./Data/input.csv')[0:20000]
    labels = pd.read_csv('./Data/labels.csv')[0:20000]

    # Load and preprocess your data here
    X = preprocess_data(input)
    labels.drop('id', axis=1, inplace=True)

    y = labels['status_group'].tolist()

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train your model
    model = Model(X_train, y_train, X_test, y_test, hyperparameters=hyperparameters)
    model.train()  # Update this method to use the provided dataset

    # Predict and evaluate the model
    predictions = model.predict() 
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

def run_optimization():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print(study.best_params)


if __name__ == "__main__":
    # args = sys.argv[1:]

    # if len(args) != 7:
    #     print('Invalid Usage!')
    #     print('python part1.py <train-input-file> <train-labels-file> <test-input-file> <numerical-preprocessing> <categorical-preprocessing> <model-type> <test-prediction-output-file>')
    #     sys.exit(0)
    # else:
    #     train_input, train_labels, test_input, num_proc, cat_proc, model_type, prediction_output = args

    #     if not num_proc in ['None', 'StandardScaler']:
    #         print('Numerical Processing must be one of the following options : [None | StandardScaler]')
    #         sys.exit(0)
    #     elif not cat_proc in ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder']:
    #         print('Categorial Process must be one of the following options : [OneHotEncoder | OrdinalEncoder | TargetEncoder]')
    #         sys.exit(0)
    #     elif not model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'MLPClassifier', 'LogisticRegression']:
    #         print("Model type must be one of the following : [RandomForestClassifier | GradientBoostingClassifier | HistGradientBoostingClassifier | MLPClassifier | LogisticRegression]")
    #         sys.exit(0)
    #     elif train_input.split('.').pop() != 'csv' or train_labels.split('.').pop() != 'csv' or test_input.split('.').pop() != 'csv' or prediction_output.split('.').pop() != 'csv':
    #         print('All files must be csv files')
    #         sys.exit(0)

    run_optimization()

'''
    RandomForestClassifier: 0.7935
        n_estimators = 344
        max_depth = 18
        min_samples_split = 2
        min_samples_leaf = 1
        max_features = sqrt

'''