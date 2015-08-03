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
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy
from test_learning import BestClassifier, Best2Classifiers, BestAndRandom

def load_imdb( path, shuffle=True, random_state = 42,
              vectorizer = CountVectorizer( min_df = 5, max_df = 1.0, binary = True)):

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
    print("done in %fs" %duration)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

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

    parameterlist = ["alpha=0.01","alpha=0.1","alpha=1.0","alpha=10.0","alpha=100.0","alpha=1000.0"]
    floderlist = ["./unc_BernoulliNB_alpha = 0.01/","./unc_BernoulliNB_alpha = 0.1/","./unc_BernoulliNB_alpha = 1.0/",
                    "./unc_BernoulliNB_alpha = 10.0/","./unc_BernoulliNB_alpha = 100.0/"]

    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
    X_tr, y_tr, X_te, y_te, indices = load_imdb("C:\\Users\\Ping\\Desktop\\aclImdb", vectorizer=vect)

    # Use the Active Learning
    # print "BestClassifier"
    # folderpath_1 = './' +args.strategies + '_' + args.classifier + '_BestClassifier/'
    #
    # if not os.path.exists(folderpath_1):
    #     os.mkdir(folderpath_1)
    #
    # learning = BestClassifier(parameterlist, floderlist, args.num_trials, args.bootstrap,  args.stepsize, args.budget, args.strategies, args.classifier)
    #
    # learning.run_trials(X_tr, y_tr, X_te, y_te, args.strategies, args.classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath_1)
    #
    # print "Best2Classifiers"
    # folderpath_2 = './' +args.strategies + '_' + args.classifier + '_Best2Classifier/'
    #
    # if not os.path.exists(folderpath_2):
    #     os.mkdir(folderpath_2)
    # learning2 = Best2Classifiers(parameterlist, floderlist, args.num_trials, args.bootstrap,  args.stepsize, args.budget, args.strategies, args.classifier)
    #
    # learning2.run_trials(X_tr, y_tr, X_te, y_te, args.strategies, args.classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath_2)

    folderpath_3 = './' +args.strategies + '_' + args.classifier + '_BestClassifier_Random/'

    if not os.path.exists(folderpath_3):
        os.mkdir(folderpath_3)

    learning3 = BestAndRandom(parameterlist, floderlist, args.num_trials, args.bootstrap,  args.stepsize, args.budget, args.strategies, args.classifier)
    learning3.run_trials(X_tr, y_tr, X_te, y_te, args.strategies, args.classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials, folderpath_3)