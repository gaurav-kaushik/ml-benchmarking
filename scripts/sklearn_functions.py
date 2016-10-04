#!/usr/bin/env python
"""----------------------------------------------------

Helper functions for scikit-learn

    : score_model
        returns mean sq err,  variance score

    : autotune_alpha
        finds alpha in range that maximizes the variance score
        returns best_alpha, best_model

    : train_without_cv
        split data (no cv) and train model

    : train_without_cv_all
        split data (no cv) and train all models in a dict (e.g. classifiers())

    : add_noisy_features
        add additional noisy features to X

    : add_noise_normal
        add normal noise to each element in X

    : plot_ROC
        plots ROC for cv in binary classification

    : regression_models
        returns dict of common regression models

    : classifiers
        returns dict of common classification models
---------------------------------------------------- """
from __future__ import division, print_function
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso, ElasticNet, \
                                    Lars, LassoLars, LogisticRegression, Perceptron
from sklearn import svm, neighbors, tree, ensemble, naive_bayes, discriminant_analysis
from sklearn.metrics import roc_curve, auc
# from sklearn.multiclass import OneVsRestClassifier
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold  # sklearn > v0.18
except:
    from sklearn.cross_validation import train_test_split, StratifiedKFold  # sklearn < v0.18

""" Score Models """

def score_model(model, X_test, y_test):
    # Mean Squared Error of prediction versus truth
    score_meansqerr = np.mean((model.predict(X_test) - y_test) ** 2)
    # Variance score (1 = perfect relationship, 0 = no linear dependence)
    score_var = model.score(X_test, y_test)
    return score_meansqerr, score_var


""" Tuning hyperparameters """

def alpha_testing(model, X_train, X_test, y_train, y_test):
    # ToDo: Fix bug in all alphas returning same values
    print("Alpha Testing:")
    print("\tDefault scores:", score_model(model, X_test, y_test))

    new_alpha, new_model = autotune_alpha(model, X_train, X_test, y_train, y_test)

    print("\tNew alpha:", new_alpha)
    print("\tTuned scores:", score_model(new_model, X_test, y_test))

    return new_model


def autotune_alpha(model, X_train, X_test, y_train, y_test):
    # Set range of alpha values
    # alphas = np.arange(alpha_min, alpha_max+alpha_step, alpha_step)
    alphas = np.logspace(-4, -1, 6)

    # Generate model for each value and score
    models = [model.set_params(alpha=a).fit(X_train, y_train) for a in alphas]
    scores = [m.score(X_test, y_test) for m in models]

    # Select best alpha and model (has max variance score)
    best_index = scores.index(max(scores))
    best_alpha, best_model = alphas[best_index], models[best_index]
    return best_alpha, best_model


""" Training Functions """

def train_without_cv_all(models, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    for k, v in models.items():
        print("\t", k, ":", score_model(v.fit(X_train, y_train), X_test, y_test))
    return models

def train_without_cv(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("\t", model, ":", score_model(model.fit(X_train, y_train), X_test, y_test))
    return model


""" Noise function """

def add_noisy_features(X, degree=100, rand_state=0):
    # Add noisy features (add features with random noise)
    n_samples, n_features = X.shape
    return np.c_[X, np.random.RandomState(rand_state).randn(n_samples, degree*n_features)]

def add_noise_normal(X, degree=1):
    # Add noise to the matrix itself
    return np.random.normal(X, degree)

""" Plots """

def plot_ROC(model, X, y, cv_fold=5):
    # Set up mean true and false positive rates
    mean_tpr, mean_fpr = 0, np.linspace(0, 1, 100)

    # Set the matplotlib colorwheel as a cycle
    colors = itertools.cycle(list(cnames.keys()))
    # Set the cross-validation fold
    skf = StratifiedKFold(cv_fold)  # list(skf.split(X,y)) returns list of len n_splits

    # loop over each split in the data and plot the ROC
    for idx, val in enumerate(zip(skf.split(X, y), colors)):
        (train, test), color = val
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])  # pull the values of X,y using indices
        fpr, tpr, _ = roc_curve(y[test], probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        plt.plot(fpr, tpr, lw=2, color=color, label='ROC fold %d (area = %0.2f)' % (idx, roc_auc))

    # Plot mean tpr for all curves
    mean_tpr /= skf.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='k', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=4)

    # Plot chance (tpr = fpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', label='Chance')

    # Axes and labels
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve ({}-fold CV)'.format(cv_fold))
    plt.legend(loc="lower right")
    plt.show()

    return

""" Generate Models """
def regression_models():
    """
    Regression Models
    """
    return {
        'lsq': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elasticnet': ElasticNet(),
        'lars': Lars(),
        'lassolars': LassoLars(),
        'bayes': BayesianRidge(),
        'logreg': LogisticRegression(),
        'perceptron': Perceptron(),

        # SVMs
        'svr_rbf': svm.SVR(kernel='rbf'),
        'svr_lin': svm.SVR(kernel='linear'),
        'svr_poly': svm.SVR(kernel='poly')
    }

def classifiers():
    """
    Classification Models
    """
    return {
        'kneighbors': neighbors.KNeighborsClassifier(),
        'svc_lin': svm.SVC(kernel='linear', probability=True),
        'svc_rbf': svm.SVC(probability=True),
        'svc_poly': svm.SVC(kernel='poly', degree=2, probability=True),
        'decision_tree': tree.DecisionTreeClassifier(),
        'random_forest': ensemble.RandomForestClassifier(),
        'adaboost': ensemble.AdaBoostClassifier(),
        'gaussian_nb': naive_bayes.GaussianNB(),
        'lin_da': discriminant_analysis.LinearDiscriminantAnalysis(),
        'quad_da': discriminant_analysis.QuadraticDiscriminantAnalysis()
    }


if __name__ == "__main__":
    print(__doc__)