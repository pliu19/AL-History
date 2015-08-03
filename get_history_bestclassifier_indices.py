__author__ = 'Ping123'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
import os
from time import time
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy

class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""

    def __init__(self, parameters = None):
        self.list_para = parameters

    def _get_indices(self, trial):

        indexofparameter = np.loadtxt(open("./unc_LogisticRegression/Index_Claasifier_Trial" + str(trial+1) + ".txt","rb"))
        list1 = ["C=0.01","C=0.1","C=1.0","C=10.0","C=100.0","C=1000.0"]
        list2 = []

        for i in range(100):
            initial = "./unc_LogisticRegression_"
            end = list1[int(indexofparameter[i])]
            filename = initial + end + "/Indices_record_Trial_" + str(trial+1) +".txt"

            temp = np.loadtxt(open(filename,"rb"),delimiter = ',',dtype ="int")
            list2.extend(temp[i].tolist())
        return list2

    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, num_trials, path,lengthoftxt=None):

        for t in range(num_trials):
            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, budget, t,path, lengthoftxt=None)

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, t, folderpath,lengthoftxt=None):
        """Helper method for running multiple trials."""

        list_indices = self._get_indices(t)

        file_name_parameter = "./unc_BernoulliNB/Index_Claasifier_Trial" + str(t + 1) + ".txt"
        parameters = np.loadtxt(open(file_name_parameter,"rb"))

        parameterslist = ["alpha=0.01","alpha=0.1","alpha=1.0","alpha=10.0","alpha=100.0"]

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

        model = None
        labels = np.unique(y_pool)

        #Loop for prediction
        ite = 0
        accuracyInTrail = []

        sum = 0
        bootstrapped = False
        while sum < budget and len(pool) >= step_size:

            if not bootstrapped:
                sum = bootstrap_size
                newIndices = list_indices[:sum]
                bootstrapped = True
            else:
                sum = sum + step_size
                newIndices = list_indices[:sum]

            newIndices = np.unique(newIndices)

            model = get_classifier(classifier_name, parameterslist[int(parameters[ite])])

            model = eval(model)

            model.fit(X_pool[newIndices], y_pool[newIndices])
            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            tempaccuracy = accuracy_score(y_test,y_pred)
            accuracyInTrail.append(tempaccuracy)
            # Save the prediction and probas

            result_prediction[:,ite] = y_pred
            result_probas[:,ite] = y_probas[:,0]

            ite = ite + 1

            if ite == 100:
                print "Length of Indices of last iterations %r " %len(newIndices)


        file_name_accuracy = folderpath +"Accuracy_record_Trial" + "_" +str(t+1)
        write_float(file_name_accuracy,accuracyInTrail)

        accu = accuracy_score(y_test, result_prediction[:,column-2])
        print "This is the %r-th trial, the accuracy is %r" %(str(t+1),accu)

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        file_name_proba = folderpath + args.strategies + "_" + "Trial" + "_" + str(t+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, result_probas, delimiter=",")

        file_name_prediction = folderpath + args.strategies + "_" + "Trial" + "_" + str(t+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, result_prediction, delimiter=",")

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
    # print type(X_test)

    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape
    print

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

    result = str(classifier) + '(' + argus + ')'
    return result



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-path', help='The path to the content file.')

    parser.add_argument('-classifier',choices=['LogisticRegression','MultinomialNB','SVC','DecisionTreeClassifier'], default='BernoulliNB',
                        help='The underlying classifier.')

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

    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))
    X_tr, y_tr, X_te, y_te, indices = load_imdb("C:\\Users\\Ping\\Desktop\\aclImdb", vectorizer=vect)

    # Use the Active Learning
    folderpath = './' +args.strategies + '_' + args.classifier + '_BestClassifier_Indices/'

    if not os.path.exists(folderpath):
        os.mkdir(folderpath)

    learning_api = LearningCurve()
    learning_api.run_trials(X_tr, y_tr, X_te, y_te, args.strategies, args.classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath)

