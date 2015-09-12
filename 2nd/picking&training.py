__author__ = 'Ping'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
import os
import re
import sys
import csv
from time import time
from zipfile import ZipFile
import matplotlib as plt

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import cross_validation
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy
from plot_analysis import plot_result


class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, train_classifier, test_classifier, bootstrap_size,  step_size, budget, num_trials, path,lengthoftxt=None):


        rows = int(budget-bootstrap_size)/step_size + 1

        result = np.zeros(shape=(rows,))

        for t in range(num_trials):

            print "trial", t+1

            result += self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, train_classifier, test_classifier, bootstrap_size,  step_size, budget, t,path, lengthoftxt=None)

        return result

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, train_classifier, test_classifier, bootstrap_size,  step_size, budget, t, folderpath,lengthoftxt=None):
        """Helper method for running multiple trials."""

        rows = len(y_test)
        column = int(budget/step_size) + 1

        pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        #Choosing strategy
        if al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        model = None
        labels = np.unique(y_pool)

        #Loop for prediction
        ite = 0

        accuracyInTrail = []

        while len(trainIndices) < budget and len(pool) >= step_size:
            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model_pick_classifier, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])


            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)
            model_pick_classifier = eval(train_classifier)

            model_pick_classifier.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            model_test_classifier = eval(test_classifier)
            model_test_classifier.fit(X_pool[trainIndices], y_pool[trainIndices])

            y_pred = model_test_classifier.predict(X_test)

            tempaccuracy = accuracy_score(y_test,y_pred)
            accuracyInTrail.append(tempaccuracy)

            ite = ite + 1
        accuracyInTrail = np.array(accuracyInTrail)

        print accuracyInTrail[0]

        return accuracyInTrail

    def get_mean_accuracy(self, folderpath, iteration, trial):
        sum = np.zeros(shape=(iteration))
        for j in range(trial):
            temp_name = folderpath+ "Accuracy_record_Trial_" + str(j+1) + ".txt"
            temp = np.loadtxt(open(temp_name,"rb"),delimiter=",")
            sum = sum + temp
        sum = sum / 10.
        file_name = folderpath + "average_accuracy"
        np.savetxt("%s.txt" %file_name, sum, delimiter=",",fmt='%10.10f')

    def plot_history_proba(self, number):
        return


def write_integer(path,array):
    array = np.asarray(array)
    np.savetxt("%s.txt" %path, array.astype(int), delimiter=",",fmt='%i')

def write_float(path,array):
    array = np.asarray(array)
    np.savetxt("%s.txt" %path, array, delimiter=",",fmt='%10.10f')

def load_imdb(path, shuffle=True, random_state=42, \
              vectorizer = CountVectorizer(min_df=5, max_df=1.0, binary=True)):

    print "Loading the imdb reviews data"

    train_neg_files = glob.glob(path+"/train/neg/*.txt")
    train_pos_files = glob.glob(path+"/train/pos/*.txt")

    train_corpus = []
    y_train = []

    for tnf in train_neg_files:
        f = open(tnf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(0)
        f.close()

    # print train_corpus

    for tpf in train_pos_files:
        f = open(tpf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(1)
        f.close()

    test_neg_files = glob.glob(path+"/test/neg/*.txt")
    test_pos_files = glob.glob(path+"/test/pos/*.txt")

    test_corpus = []

    y_test = []

    for tnf in test_neg_files:
        f = open(tnf, 'r')
        test_corpus.append(f.read())
        y_test.append(0)
        f.close()

    for tpf in test_pos_files:
        f = open(tpf, 'r')
        test_corpus.append(f.read())
        y_test.append(1)
        f.close()

    print "Data loaded."

    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    t0 = time()

    X_train = vectorizer.fit_transform(train_corpus)


    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_train.shape
    print

    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()

    X_test = vectorizer.transform(test_corpus)

    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))

        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]

        indices = np.random.permutation(len(y_test))

        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
    else:
        train_corpus_shuffled = train_corpus
        test_corpus_shuffled = test_corpus

    return X_train, y_train, X_test, y_test, indices


def get_classifier(classifier, argus):
    result = classifier + '(' + argus + ')'
    return result

def Lastcross(array,lengthoftest):

    last_cross = []
    for j in range(lengthoftest):
        temp = array[j]

        if temp[0]>0.5:
            lastpred = 0
        else:
            lastpred = 1

        length = len(temp)
        lastcross = 0
        count=0
        for i in range(length-1):
            if temp[i]>0.5:
                flag = 0
            else:
                flag = 1
            if flag != lastpred:
                count += 1
                lastcross = i
            lastpred = flag
        last_cross.append(lastcross)

    return last_cross

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', default = "C:\\Users\\Ping\\Desktop\\ibn_sina.zip", help='The path to the content file.')

    parser.add_argument('-filename', default = "ibn_sina", help='The name of file.')

    parser.add_argument('-feature_scaling', default = True)

    parser.add_argument('-classifier',choices=['LogisticRegression','MultinomialNB','SVC','DecisionTreeClassifier','BernoulliNB'], default='LogisticRegression',
                        help='The underlying classifier.')

    parser.add_argument("--picking_arguments", default="C=0.1",
                        help="Represents the arguments that will be passed to the classifier (default: '').")
    parser.add_argument("--training_arguments", default=["C=0.01","C=0.1","C=1.0","C=10.0","C=100.0","C=1000.0"],
                        help="Represents the argumenxts that will be passed to the classifier (default: '').")

    parser.add_argument("-nt", "--num_trials", type=int, default=5, help="Number of trials (default: 10).")

    parser.add_argument("-st", "--strategies", choices=['qbc', 'rand','unc'], nargs='*',default='unc',
                        help="Represent a list of strategies for choosing next samples (default: unc).")

    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')

    parser.add_argument("-b", '--budget', default=310, type=int,
                        help='Sets the budget (default: 2000).')

    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')

    parser.add_argument("-cv", type=int, default=5,
                        help="Number of folds for cross validation. Works only if a single dataset is loaded (default: 10).")

    args = parser.parse_args()

    # vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))

    X,y = load_data(args.path)

    if args.feature_scaling:
        X = scale(X)

    skf = StratifiedKFold(y, n_folds=args.cv, shuffle=True, random_state=42)

    row = int((args.budget-args.bootstrap)/args.stepsize + 1)

    file_folder = './' + args.filename + '/'
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)

    for i in args.training_arguments:
        print "################################################################"
        print "The arguments of this test:"
        print "     Dataset: %s" %args.filename
        print "     Classifier: %s, picking: %s, training: %s" %(args.classifier, args.picking_arguments, i)
        print "     Strategy: %s " % args.strategies
        print "     budget: %s    bootstrap: %s    stepsize: %s" %(args.budget, args.bootstrap, args.stepsize)
        print "################################################################"

        combine_picking_classifier = get_classifier(args.classifier, args.picking_arguments)
        combine_training_classifier = get_classifier(args.classifier,i)
        folderpath = file_folder + args.strategies + '_' + args.classifier + '_(picking_' + args.picking_arguments + ')_(training_' + i + ')/'

        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        average_result = np.zeros(row,)

        for pool, test in skf:
            learning_api = LearningCurve()
            average_result += learning_api.run_trials( X[pool], y[pool], X[test], y[test], args.strategies,combine_picking_classifier, combine_training_classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath)

        average_result = average_result/ (args.cv*args.num_trials)

        file_name = folderpath + "average_accuracy"
        np.savetxt("%s.txt" %file_name, average_result, delimiter=",",fmt='%10.10f')






