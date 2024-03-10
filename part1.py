import pandas as pd 
import numpy as np 
import sys 

from datetime import datetime

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

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()


class Model():

    def __init__(self, args, maxiters):
        ''' 
            Constructor 
            @param (args) : command line arguments as list 
            @param (maxiters) : max iterations for gradient descent 
        '''

        # Extract command line arguments 
        train_input, train_labels, test_input, num_proc, cat_proc, model_type, self.prediction_output = args

        # Read and prerocess train inputs, test inputs, and train labels 
        self.input = self.preprocess_data(pd.read_csv(train_input))
        self.labels = (pd.read_csv(train_labels).drop('id', axis=1))['status_group'].tolist()

        self.test_input = pd.read_csv(test_input)
        self.id_col = self.test_input['id'].tolist()
        self.test_input = self.preprocess_data(self.test_input)

        # Configure 
        num_processor = None 
        if (num_proc == 'StandardScaler'):
            num_processor = StandardScaler()

        # Configure which features are numerical and which are not
        num_features = ['amount_tsh', 'gps_height', 'longitude','latitude', 'population', 'construction_year','years_since_construction']
        cat_features = [feature for feature in self.input.columns if feature not in num_features]

        # Initialize categorical feature pre-processor 
        cat_preprocessor = self.get_cat_preprocessor(cat_proc, num_features, cat_features, num_processor)

        # Initialize classifier model 
        self.model = self.get_model(model_type, cat_preprocessor, cat_proc, maxiters)


    def get_top_k_frequent(self, df, column_name, k=100):
        top_k_frequent = df[column_name].value_counts().head(k).index.tolist()
        return top_k_frequent
    
    def replace_col_with_topk(self, df, column_name, top_k):
        # Identify values in `column_name` that are not in `top_k`
        condition = ~df[column_name].isin(top_k)
        
        # Replace values in `column_name` based on the condition
        df.loc[condition, column_name] = 'other'

    def preprocess_data(self, df):
        '''
            Pre-processed input data - performs feature engineering 
            @param (df) : data frame representing inputs 
        '''
        now = datetime.now()

        # Modify date_recorded column to days from now since days recorded
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        year = df['date_recorded'].dt.year 
        df['years_since_construction'] = (year - df['construction_year'])

        drop_columns = ['date_recorded', 'num_private', 'id', 'wpt_name', 'subvillage']
        df.drop(drop_columns, axis=1, inplace=True)

        reduce_columns = ['ward','funder','installer', 'scheme_name']

        for column in reduce_columns:
            top_k = self.get_top_k_frequent(df, column)
            self.replace_col_with_topk(df, column, top_k)

        return df


    def get_cat_preprocessor(self, cat_proc, num_features, cat_features, num_processor):
        '''
            For initializing the preprocessor for encoding categorical features 
            @param (cat_proc) : String indicating type of encoding method 
            @param (num_features) : list of string names of all numerical features 
            @param (cat_features) : list of string names of categorical features 
            @param (num_processor) : the numerical feature processor (None or StandardScaler())
        '''
        
        if cat_proc == 'OneHotEncoder':
            return ColumnTransformer(
                        transformers=[
                            ('num', num_processor, num_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                        ]
                    )
        elif cat_proc == 'OrdinalEncoder':
            return ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),  # Impute numerical features
                            ('scaler', num_processor)
                        ]), num_features),
                        ('cat', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical features
                            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                        ]), cat_features)
                    ]
                )
        else :
            return ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),  # Impute numerical features
                            ('scaler', num_processor)
                        ]), num_features),
                        ('cat', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical features
                            ('encoder', TargetEncoder())
                        ]), cat_features)
                    ]
                )

    def get_model(self, model_type, cat_preprocessor, cat_type, maxiters):
        ''' 
            For initializing classification model 
            @param (model_type) : string indicating type of model 
            @param (cat_preprocessor) : feature encoder 
            @param (cat_type) : string indicating feature encoding type 
            @param (maxiters) : max iterations for gradient descent
        '''
        if model_type == 'HistGradientBoostingClassifier':
            if cat_type == 'OneHotEncoding':
                return Pipeline(steps=[('preprocessor', cat_preprocessor),
                                       ('to_dense', DenseTransformer()),
                                       ('classifier', HistGradientBoostingClassifier(max_iter=maxiters))])
            return Pipeline(steps=[('preprocessor', cat_preprocessor),
                                       ('classifier', HistGradientBoostingClassifier(max_iter=maxiters))])
        elif model_type == 'GradientBoostingClassifier':
            return  Pipeline(steps=[('preprocessor', cat_preprocessor),
                        ('classifier', GradientBoostingClassifier(max_iter=maxiters))])
        elif model_type == 'LogisticRegression':
            return Pipeline(steps=[('preprocessor', cat_preprocessor),
                        ('classifier', LogisticRegression(max_iter=maxiters))])
        elif model_type == 'RandomForestClassifier':
            return Pipeline(steps=[('preprocessor', cat_preprocessor),
                        ('classifier', RandomForestClassifier(max_iter=maxiters))])
        else :
            return Pipeline(steps=[('preprocessor', cat_preprocessor),
                        ('classifier', MLPClassifier(max_iter=maxiters))])
        
    
    def KFoldValidation(self):
        '''
            Performs K-Fold Validation 
        '''
        print('TRAINING MODEL...\n')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, self.input, self.labels, cv=kf, scoring='accuracy')
        print(f'K-Fold Cross Validation [K=5] Accuracy')
        print('     Per-fold Accuracy :', scores)
        print('     Average Accuracy :', np.mean(scores))

    def predict(self):
        '''
            Performs prediction using the test inputs and saves predictions to specified CSV file 
        '''
        self.model.fit(self.input, self.labels)
        predictions = self.model.predict(self.test_input)

        predictions = zip(self.id_col, predictions)
        result_df = pd.DataFrame(predictions, columns=['id','status_group'])
        result_df.to_csv(self.prediction_output, index=False)

        print(f'\nPredictions saved to [{self.prediction_output}]')


def main(args):
    model = Model(args, 1000) 
    model.KFoldValidation()
    model.predict()

    print()

if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) != 7:
        print('Invalid Usage!')
        print('python part1.py <train-input-file> <train-labels-file> <test-input-file> <numerical-preprocessing> <categorical-preprocessing> <model-type> <test-prediction-output-file>')
        sys.exit(0)
    else:
        train_input, train_labels, test_input, num_proc, cat_proc, model_type, prediction_output = args

        if not num_proc in ['None', 'StandardScaler']:
            print('Numerical Processing must be one of the following options : [None | StandardScaler]')
            sys.exit(0)
        elif not cat_proc in ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder']:
            print('Categorial Process must be one of the following options : [OneHotEncoder | OrdinalEncoder | TargetEncoder]')
            sys.exit(0)
        elif not model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'MLPClassifier', 'LogisticRegression']:
            print("Model type must be one of the following : [RandomForestClassifier | GradientBoostingClassifier | HistGradientBoostingClassifier | MLPClassifier | LogisticRegression]")
            sys.exit(0)
        elif train_input.split('.').pop() != 'csv' or train_labels.split('.').pop() != 'csv' or test_input.split('.').pop() != 'csv' or prediction_output.split('.').pop() != 'csv':
            print('All files must be csv files')
            sys.exit(0)

        main(args)
        

