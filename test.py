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
                            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
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
    # # Define the hyper-parameter search space
    # n_estimators = trial.suggest_int('n_estimators', 10, 500)
    # max_depth = trial.suggest_int('max_depth', 2, 32)

    # # Hyperparameters to pass into your model
    # hyperparameters = {
    #     'n_estimators': n_estimators,
    #     'max_depth': max_depth,
    #     # Include other hyperparameters here
    # }

    # Existing hyper-parameter suggestions
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    
    # New hyper-parameter suggestions
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Hyperparameters dictionary
    hyperparameters = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
    }

    input = pd.read_csv('./Data/input.csv')[0:20000]
    labels = pd.read_csv('./Data/labels.csv')[0:20000]

    # Load and preprocess your data here
    # For demonstration, we assume you have a function to get your processed data
    X = preprocess_data(input)
    labels.drop('id', axis=1, inplace=True)

    y = labels['status_group'].tolist()

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train your model
    model = Model(X_train, y_train, X_test, y_test, hyperparameters=hyperparameters)
    model.train()  # Update this method to use the provided dataset

    # Predict and evaluate the model
    predictions = model.predict()  # Ensure this method returns the predicted labels
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

# Entry point to run the script
if __name__ == "__main__":
    run_optimization()