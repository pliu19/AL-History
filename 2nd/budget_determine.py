__author__ = 'Ping'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
import os
from zipfile import ZipFile
from sklearn import cross_validation
import csv
from time import time
import matplotlib as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy

class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, accuracy, num_trials):

        accumulative_budget = 0
        for t in range(num_trials):
            print "trial", t+1
            accumulative_budget += self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, accuracy, t)

        return accumulative_budget

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, accuracy, t):
        """Helper method for running multiple trials."""

        pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        #Choosing strategy
        if al_strategy == 'qbc':
            active_s = QBCStrategy(classifier=classifier_name)
        elif al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        model = None
        labels = np.unique(y_pool)

        #Loop for prediction

        current_accuracy = 0

        budget = 0

        while accuracy >= current_accuracy and len(pool) >= step_size:
            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
                budget += bootstrap_size
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

                budget += step_size

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)
            model = eval(classifier_name)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            y_pred = model.predict(X_test)
            current_accuracy = accuracy_score(y_test,y_pred)

            print current_accuracy

        print "The budget of this trial is %s" %budget
        return budget

def load_data(dataset1, dataset2=None, make_dense=False):
    """Loads the dataset(s).
    Can handle zip files.
    If the data file extension is csv, it reads a csv file.
    Then, the last column is treated as the target variable.
    Otherwise, the data files are assumed to be in svmlight/libsvm format.

    **Parameters**

    * dataset1 (*str*) - Path to the file of the first dataset.
    * dataset2 (*str or None*) - If not None, path to the file of second dataset
    * make_dense (*boolean*) - Whether to return dense matrices instead of sparse ones (Note: data from csv files will always be treated as dense)

    **Returns**

    * (X_pool, X_test, y_pool, y_test) - Pool and test files if two files are provided
    * (X, y) - The single dataset

    """

    def _get_extensions(dataset1, dataset2):
        first_extension = dataset1[dataset1.rfind('.')+1:]
        second_extension = None
        if dataset2 is not None:
            second_extension = dataset2[dataset2.rfind('.')+1:]

        return first_extension, second_extension

    # Test if these are zipped files

    fe, se = _get_extensions(dataset1, dataset2)

    if se and fe != se:
        raise ValueError("Cannot mix and match different file formats")

    iz_zip = fe == 'zip'

    # Open the files and test if these are csv
    dataset1_file = None
    dataset2_file = None
    is_csv = False

    if iz_zip:
        my_zip_dataset1 = ZipFile(dataset1)
        inside_zip_dataset1 = my_zip_dataset1.namelist()[0] # Assuming each zip contains a single file
        inside_zip_dataset2 = None
        dataset1_file = my_zip_dataset1.open(inside_zip_dataset1)
        if dataset2 is not None:
            my_zip_dataset2 = ZipFile(dataset2)
            inside_zip_dataset2 = my_zip_dataset2.namelist()[0] # Assuming each zip contains a single file
            dataset2_file = my_zip_dataset2.open(inside_zip_dataset2)
        inside_fe, inside_se = _get_extensions(inside_zip_dataset1, inside_zip_dataset2)
        if inside_se and inside_fe != inside_se:
            raise ValueError("Cannot mix and match different file formats")

        is_csv = inside_fe == 'csv'
    else:

        dataset1_file = open(dataset1, 'r')
        if dataset2 is not None:
            dataset2_file = open(dataset2, 'r')

        is_csv = fe == 'csv'

    if dataset2 is not None:
        if is_csv:
            X_pool, y_pool = load_csv(dataset1_file)
            X_test, y_test = load_csv(dataset2_file)
        else:
            X_pool, y_pool = load_svmlight_file(dataset1_file)
            _, num_feat = X_pool.shape
            X_test, y_test = load_svmlight_file(dataset2_file, n_features=num_feat)
            if make_dense:
                X_pool = X_pool.todense()
                X_test = X_test.todense()

        le = LabelEncoder()
        y_pool = le.fit_transform(y_pool)
        y_test = le.transform(y_test)

        dataset1_file.close()
        dataset2_file.close()

        return (X_pool, X_test, y_pool, y_test)

    else:

        if is_csv:
            X, y = load_csv(dataset1_file)
        else:
            X, y = load_svmlight_file(dataset1_file)
            if make_dense:
                X = X.todense()

        le = LabelEncoder()
        y = le.fit_transform(y)

        dataset1_file.close()


        return X, y

def load_csv(dataset_file):
    X=[]
    y=[]
    csvreader = csv.reader(dataset_file, delimiter=',')
    next(csvreader, None)#skip names
    for row in csvreader:
        X.append(row[:-1])
        y.append(row[-1])
    X=np.array(X, dtype=float)
    y=np.array(y)
    return X, y

def get_classifier(classifier, argus):
    result = classifier + '(' + argus + ')'
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', default= 'C:\\Users\\Ping\\Desktop\\nova.zip' ,help='The path to the content file.')

    parser.add_argument('-classifier',choices=['LogisticRegression','MultinomialNB','SVC','DecisionTreeClassifier'], default='LogisticRegression',
                        help='The underling classifier.')

    parser.add_argument("-a","--arguments", default="",
                        help="Represents the arguments that will be passed to the classifier (default: '').")

    parser.add_argument("-nt", "--num_trials", type=int, default=5, help="Number of trials (default: 10).")

    parser.add_argument("-st", "--strategies", choices=['qbc', 'rand','unc'], nargs='*',default='rand',
                        help="Represent a list of strategies for choosing next samples (default: unc).")

    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')

    parser.add_argument("-accuracy", default = 0.959207283180154, type = float)
    # parser.add_argument("-b", '--budget', default=5000, type=int,
    #                     help='Sets the budget (default: 2000).')

    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')

    parser.add_argument("-cv", type=int, default=5,
                        help="Number of folds for cross validation. Works only if a single dataset is loaded (default: 10).")

    args = parser.parse_args()

    X, y = load_data(args.path)

    skf = StratifiedKFold(y, n_folds=args.cv, shuffle=True, random_state=42)

    combine_classifier = get_classifier(args.classifier, args.arguments)

    result = 0
    for pool, test in skf:

        learning_api = LearningCurve()
        result += learning_api.run_trials( X[pool], y[pool], X[test], y[test], args.strategies, combine_classifier, args.bootstrap, args.stepsize, args.accuracy, args.num_trials)

    result = float(result / (args.cv * args.num_trials))

    print result

