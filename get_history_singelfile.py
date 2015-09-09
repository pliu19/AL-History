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
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, num_trials, path,lengthoftxt=None):

        self.history_probas = { }
        self.history_preditcion = { }

        for t in range(num_trials):
            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, budget, t,path, lengthoftxt=None)



    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, t, folderpath,lengthoftxt=None):
        """Helper method for running multiple trials."""

        rows = len(y_test)
        column = int(budget/step_size) + 1

        result_prediction = np.zeros(shape=(rows, column))
        result_probas = np.zeros(shape=(rows, column))

        if len(y_pool) > 10000:
            rs = np.random.RandomState(t)
            indices = rs.permutation(len(y_pool))
            pool = set(indices[:10000])
        else:
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
        ite = 0
        while len(trainIndices) < budget and len(pool) >= step_size:
            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)
            model = eval(classifier_name)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            result_prediction[:,ite] = y_pred
            result_probas[:,ite] = y_probas[:,0]

            ite = ite + 1

        accu = accuracy_score(y_test, result_prediction[:,column-2])
        print "This is the %r-th trial, the accuracy is %r" %(str(t+1),accu)

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        file_name_proba = folderpath + args.strategies + "_" + "Trial" + "_" + str(t+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, result_probas, delimiter=",")

        file_name_prediction = folderpath + args.strategies + "_" + "Trial" + "_" + str(t+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, result_prediction, delimiter=",")

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
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=(1./2.), random_state=42)
        print "The length of X_train(X_test) is %r" %len(y_train)
        return X_train, X_test, y_train, y_test



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


def table_of_cross(array, lengthoftxt):
    output = np.zeros(shape=(25000,4))
    for j in range(25000):
        temp = array[j]
        lastpred=temp[0]
        length=len(temp)
        lastcross=0
        count=0
        for i in range(length-1):
            if temp[i] != lastpred:
                count += 1
                lastcross = i
            lastpred=temp[i]
        if temp[length-2]==temp[length-1]:
            correct = 1
        else:
            correct = 0
        output[j][0] = count
        output[j][1] = lastcross
        output[j][2] = lengthoftxt[j]
        output[j][3] = correct
    return output

def get_classifier(classifier, argus):

    result = classifier + '(' + argus + ')'
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', help='The path to the content file.')

    parser.add_argument('-classifier',choices=['LogisticRegression','MultinomialNB','SVC','DecisionTreeClassifier'], default='SVC',
                        help='The underlying classifier.')
    parser.add_argument("-a","--arguments", default="kernel='poly',degree=2, C=0.1, probability=True",
                        help="Represents the arguments that will be passed to the classifier (default: '').")
    parser.add_argument("-nt", "--num_trials", type=int, default=10, help="Number of trials (default: 10).")

    parser.add_argument("-st", "--strategies", choices=['qbc', 'rand','unc'], nargs='*',default='rand',
                        help="Represent a list of strategies for choosing next samples (default: unc).")

    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')
    parser.add_argument("-b", '--budget', default=5000, type=int,
                        help='Sets the budget (default: 2000).')
    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')
    # parser.add_argument("-sp", '--subpool', default=None, type=int,
    #                     help='Sets the sub pool size (default: None).')
    args = parser.parse_args()

    X_tr, X_te,y_tr, y_te = load_data("./calhousing.zip")

    # Directly use the classifier and calculate the accuracy
    combine_classifier = get_classifier(args.classifier, args.arguments)

    model = eval(combine_classifier)
    temp = MinMaxScaler()
    temp.fit(X_tr,y_tr)
    model.fit(X_tr,y_tr)
    direct = model.predict(X_te)
    directaccuracy = accuracy_score(y_te,direct)
    print "Directly use the classifier, the accuracy is %r" %directaccuracy

    # Use the Active Learning
    folderpath = './' +args.strategies + '_' + args.classifier + '_' + args.arguments + '/'
    os.mkdir(folderpath)

    learning_api = LearningCurve()
    learning_api.run_trials(X_tr, y_tr, X_te, y_te, args.strategies, combine_classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath)












