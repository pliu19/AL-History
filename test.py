__author__ = 'Ping123'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
import os
import re
import sys
from time import time
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy
from plot_analysis import plot_result


class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, train_classifier, test_classifier, bootstrap_size,  step_size, budget, num_trials, path,lengthoftxt=None):

        self.history_probas = {  }
        self.history_preditcion = {  }

        for t in range(num_trials):
            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, train_classifier, test_classifier, bootstrap_size,  step_size, budget, t,path, lengthoftxt=None)

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, train_classifier, test_classifier, bootstrap_size,  step_size, budget, t, folderpath,lengthoftxt=None):
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
        indicesInTrail = []
        accuracyInTrail = []

        while len(trainIndices) < budget and len(pool) >= step_size:
            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model_train_classifier, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            indicesInTrail.append(newIndices)
            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)
            model_train_classifier = eval(train_classifier)

            model_train_classifier.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            model_test_classifier = eval(test_classifier)
            model_test_classifier.fit(X_pool[trainIndices], y_pool[trainIndices])
            y_probas = model_test_classifier.predict_proba(X_test)
            y_pred = model_test_classifier.predict(X_test)

            tempaccuracy = accuracy_score(y_test,y_pred)
            accuracyInTrail.append(tempaccuracy)
            # Save the prediction and probas

            result_prediction[:,ite] = y_pred
            result_probas[:,ite] = y_probas[:,0]

            ite = ite + 1

        file_name_indices = folderpath + "Indices_record_Trial" + "_" + str(t+1)
        write_integer(file_name_indices,indicesInTrail)
        file_name_accuracy = folderpath +"Accuracy_record_Trial" + "_" +str(t+1)
        write_float(file_name_accuracy,accuracyInTrail)

        accu = accuracy_score(y_test, result_prediction[:,column-2])
        print "This is the %r-th trial, the accuracy is %r" %(str(t+1),accu)

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        file_name_proba = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, result_probas, delimiter=",")

        file_name_prediction = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, result_prediction, delimiter=",", fmt='%i')

    def get_mean_cross(self, folder, strategy, row_file, column_file, trial):
        #row_file is the # of testdata, column_file is the # of iteration + 1

        mean_probal = np.zeros(shape=(row_file,column_file))
        for i in range(trial):
            path = folder + strategy +"_Trial_" + str(i+1)+ "_proba.csv"
            temp = np.loadtxt (open(path,"rb"), delimiter=",")
            mean_probal += temp

        mean_probal = mean_probal / float(trial)

        file_name_mean_prediction = folder + strategy + "_Mean_proba"
        np.savetxt("%s.csv" %file_name_mean_prediction, mean_probal, delimiter=",")

        last_cross_mean = Lastcross(mean_probal,row_file)
        last_cross_mean = np.array(last_cross_mean)

        file_name_lastcross_mean = folder + strategy + "_Mean_Lastcross"
        np.savetxt("%s.txt" %file_name_lastcross_mean, last_cross_mean.astype(int), delimiter=",",fmt='%i')

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

def get_lengthoftext(indices,path):
    test_neg_files = glob.glob("C:\\Users\\Ping\\Desktop\\aclImdb\\test\\neg\\*.txt")
    test_pos_files = glob.glob("C:\\Users\\Ping\\Desktop\\aclImdb\\test\\pos\\*.txt")
    length = []
    for tnf in test_neg_files:
        f = open(tnf, 'r')
        context = f.read()
        words = context.split()
        number = len(words)
        length.append(number)
        f.close()

    for tpf in test_pos_files:
        f = open(tpf,'r')
        context = f.read()
        words = context.split()
        number = len(words)
        length.append(number)
        f.close()

    length = np.array(length)
    length = length[indices]

    return length

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', default = "C:\\Users\\Ping\\Desktop\\aclImdb", help='The path to the content file.')
 
    parser.add_argument('-classifier',choices=['LogisticRegression','MultinomialNB','SVC','DecisionTreeClassifier','BernoulliNB'], default='LogisticRegression',
                        help='The underlying classifier.')
    parser.add_argument("--train_arguments", default="C=1000.0",
                        help="Represents the arguments that will be passed to the classifier (default: '').")
    parser.add_argument("--test_arguments", default=["C=0.01","C=0.1","C=1.0","C=10.0","C=100.0","C=1000.0"],
                        help="Represents the argumenxts that will be passed to the classifier (default: '').")

    parser.add_argument("-nt", "--num_trials", type=int, default=10, help="Number of trials (default: 10).")

    parser.add_argument("-st", "--strategies", choices=['qbc', 'rand','unc'], nargs='*',default='unc',
                        help="Represent a list of strategies for choosing next samples (default: unc).")

    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')
    parser.add_argument("-b", '--budget', default=1000, type=int,
                        help='Sets the budget (default: 2000).')
    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')

    args = parser.parse_args()



    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
    X_tr, y_tr, X_te, y_te, indices = load_imdb(args.path, vectorizer=vect)
    # lengthoftxt = get_lengthoftext(indices,"C:\\Users\\Ping\\Desktop\\aclImdb")

    # Directly use the classifier and calculate the accuracy
    # folderpath = './' +args.strategies + '_' + args.classifier + '_(train:' + args.train_arguments + ')_(test:' +args.test_arguments + ')/'
    # if not os.path.exists(folderpath):
    #     os.mkdir(folderpath)

    for i in args.test_arguments:
        print "################################################################"
        print "The arguments of this test:"
        print "     Classifier: %s, train: %s, test: %s" %(args.classifier, args.train_arguments, i)
        print "     Strategy: %s " % args.strategies
        print "     budget: %s    bootstrap: %s    stepsize: %s" %(args.budget, args.bootstrap, args.stepsize)
        print "################################################################"
        combine_train_classifier = get_classifier(args.classifier, args.train_arguments)
        combine_test_classifier = get_classifier(args.classifier,i)
        folderpath = './' +args.strategies + '_' + args.classifier + '_(train_' + args.train_arguments + ')_(test_' + i + ')/'

        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        learning_api = LearningCurve()
        learning_api.run_trials(X_tr, y_tr, X_te, y_te, args.strategies,combine_train_classifier, combine_test_classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath)


    # row = len(y_te)
    # column = (args.budget - args.bootstrap)/ args.stepsize + 2
    # learning_api.get_mean_cross(folderpath,args.strategies, row, column, args.num_trials)

