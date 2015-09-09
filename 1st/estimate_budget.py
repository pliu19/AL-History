from zipfile import ZipFile
import csv
import argparse
import numpy as np
from collections import defaultdict

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


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

    parser.add_argument('-path', default = "C:\\Users\\Ping\\Desktop\\",
                        help='The path to the content file.')

    parser.add_argument('-datalist', default = ['sylva'],
                        help='Deal with a set of dataset to determine the budget.')
    #'calhousing','zebra', 'nova', 'orange', 'letterO', 'letterAM', 'kdd99_10perc','ibn_sina'
    parser.add_argument("-c","--classifier", choices=['KNeighborsClassifier', 'LogisticRegression', 'SVC', 'BernoulliNB',
                        'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GaussianNB', 'MultinomialNB'],
                        default='LogisticRegression', help="Represents the classifier that will be used (default: MultinomialNB) .")

    parser.add_argument("-a","--arguments", default=['C=0.01', 'C=0.1', 'C=1.0', 'C=10.0', 'C=100.0','C=1000.0'],
                        help="Represents the arguments that will be passed to the classifier.")

    parser.add_argument("-sd", '--sdata', type=str, default='',
                        help='Single file that contains the data. Cross validation will be performed (default: None).')

    parser.add_argument("-cv", type=int, default=10,
                        help="Number of folds for cross validation. Works only if a single dataset is loaded (default: 10).")

    args = parser.parse_args()

    print "There are %s datasets in total." % len(args.datalist)
    performances = defaultdict(lambda: [])

    for data in args.datalist:
        print "This is data: %s" %data

        path = args.path + data + '.zip'
        X, y = load_data(path, None)
        print "     Load data completely"

        skf = StratifiedKFold(y, n_folds=args.cv, shuffle=True, random_state=42)

        length = len(args.arguments)
        each_accuracy = []
        for argu in args.arguments:

            print "     For arguments: %s" %argu
            counter = 0
            accuracy = []
            for pool, test in skf:
                counter += 1
                print counter
                combine_classifier = get_classifier(args.classifier, argu)
                model = eval(combine_classifier)
                model.fit(X[pool], y[pool])
                result = model.predict(X[test])
                directaccuracy = accuracy_score(y[test], result)
                accuracy.append(directaccuracy)
                print "     Directly use the classifier, the accuracy is %r" %directaccuracy
            average_accuracy = np.mean(accuracy)
            print "     The average of accuracy is %s" %average_accuracy
            each_accuracy.append(average_accuracy)
        performances[data] = each_accuracy

    print performances

    with open('mycsvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, performances.keys())
        w.writeheader()
        w.writerow(performances)







