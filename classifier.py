import argparse
from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randrange

from sklearn.metrics import accuracy_score

from GaussianNB import NaiveBayes
from Discritizer import EqualWidthDiscretiser

'''
Split dataset into train and test datasets
'''
def train_test_split(x: pd.DataFrame, y: pd.Series, test_size = 0.25, random_state = None):

    x_test = x.sample(frac = test_size, random_state = random_state)
    y_test = y[x_test.index]

    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, x_test, y_train, y_test

'''
Replace all ? values by NaN
Fill NaN values with mode or mean for categorical columns
There is no null values for numerical columns
'''
def clean_dataset(df: pd.DataFrame, use_mode=True) -> pd.DataFrame:
    for col in df.columns:
        df[col].replace('?', np.NaN, inplace=True)
    
    for df2 in [df]:
        if use_mode:
            df2['workclass'].fillna(df['workclass'].mode()[0], inplace=True)
            df2['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
            df2['native_country'].fillna(df['native_country'].mode()[0], inplace=True)
    return df

'''
Perform a transformation on categorical columns.
Values of these columns beacome new columns.
'''
def transform_categories(categorical_values, df: pd.DataFrame):
    df = pd.get_dummies(data=df, columns=categorical_values)
    return df

def perf_measure(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]=='<=50K':
           TP += 1
        if y_pred[i]=='<=50K' and y_true[i]!=y_pred[i]:
           FP += 1
        if y_true[i]==y_pred[i]=='>50K':
           TN += 1
        if y_pred[i]=='>50K' and y_true[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

def my_accuracy_score(y_true: list, y_pred: list) -> float:
    TP, FP, TN, FN = perf_measure(y_true, y_pred)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    return accuracy

# def k_fold(data, bin=10):
#     fold_len = len(data) // bin
#     rest = len(data) % bin
#     fold = []
#     train = []
#     test = []
#     pos = 0

#     for i in range(fold_len):
#         fold.append([])

#     for i in range(fold_len):
#         bin_i = 0
#         for val in data[pos:]:
#             pos += 1
#             fold[i].append(val)
#             if bin_i == fold_len:
#                 break
    

#     cross_val={'train': train, 'test': test}
#     for i, testi in enumerate(fold):
#         train.append(fold[:i] + fold[i+1:])
#         test.append(testi)
#     return cross_val







from random import seed
from random import randrange
 
'''
Split dataset into k folds
'''
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split







def rmse(y_true: list, y_pred: list):
    m = len(y_true)
    RMSE = np.sqrt(1/(2*m) * np.sum((y_pred - y_true)**2))
    print("Root Mean Square Error:", RMSE)
    return RMSE

# def normalization(df: pd.DataFrame, numerical):
#     for col in numerical:
#         min = df[col].min()
#         max = df[col].max()
#         df[col] = df[col].apply(lambda x: round((x - min) / (max - min), 2))
#     return df

def main(filename, args):
    df = pd.read_csv(filename)

    # clean dataset
    df = clean_dataset(df, args.use_mode)

    # # create X, y dataframe
    # X = df.drop(['income'], axis=1)
    # y = df['income']

    # set categorical list
    categorical = [var for var in df.columns if df[var].dtype=='O' and var != 'income']

    # transform caterogical values into columns
    X = transform_categories(categorical, df)
    columns_x = X.columns

    seed(1)
    nb_folds = 10
    folds = cross_validation_split(X.to_numpy(), nb_folds)

    i_test = nb_folds
    for fold_index in range(len(folds)):
        i_test = i_test - 1

        train_set = list()
        test_set = list()
        for i in range(nb_folds):
            if i == i_test:
                test_set = test_set + folds[i]
            else:
                train_set = train_set + folds[i]

        df = pd.DataFrame(train_set, columns=columns_x)
        # print('train_set: ', df)
        X_train = df.drop(['income'], axis=1)
        y_train = df['income']

        df2 = pd.DataFrame(test_set, columns=columns_x)
        X_test = df2.drop(['income'], axis=1)
        y_test = df2['income']

        # print(X_train.columns)
        # print(X_test.columns)

        # create categorical, numerical and columns list
        categorical = [var for var in X_train.columns if X_train[var].dtype=='O']
        numerical = [var for var in X_train.columns if X_train[var].dtype!='O']
        columns = X_train.columns

        if not args.use_mode:
            for df2 in [df]:
                for cat in categorical:
                    df2[cat].fillna(df[cat].mean(axis=0), inplace=True)

        if args.discretize == True:
            # transform numerical values unsing equibins
            disc = EqualWidthDiscretiser(bins=10, variables=numerical)
            disc.fit(X_train)
            X_train = disc.transform(X_train)
            X_test = disc.transform(X_test)

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # create naive bayes an train it
        gnb = NaiveBayes()
        gnb.fit(X_train, y_train, columns)

        # predict test dataset
        y_pred = gnb.predict(X_test)

        # verify accuracy
        print(np.unique(y_pred))
        print('my_accuracy_score: {0:0.4f}'. format(my_accuracy_score(np.array(y_test), np.array(y_pred))))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--discretize', default=True, action=argparse.BooleanOptionalAction, help='discretize numerical values using equal bins')
    parser.add_argument('--use_mode', default=True, action=argparse.BooleanOptionalAction, help='use mode or mean to fill NaN data')
    args = parser.parse_args()

    main('./adult.csv', args)
    main('./test.csv', args)

# OPTION 1: discretization => use bins with equi-width
# OPTION 2: Gaussian distribution => P(age=20 | <=50K)