from __future__ import print_function
import argparse
import numpy as np
from sklearn_functions import *
from sklearn.datasets import load_diabetes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nf", "--noisy-features", action='store_true', help="Noisy features", required=False)
    parser.add_argument("-nn", "--noise-normal", action='store_true', help="Normal noise", required=False)
    parser.add_argument("-cv", "--cross-validation", type=int, default=5, help="Cross-validation factor", required=False)
    args = vars(parser.parse_args())
    noise_normal = args['noise_normal']
    noisy_features = args['noisy_features']
    args_cv = args['cross_validation']

    # Set random seed
    np.random.seed(0)

    """ Regression """
    dataset = load_diabetes()  # regression example
    X_regr, y_regr = dataset.data, dataset.target

    """ Generate regression models w/o cross validation """
    print("Regression Models:")
    regr_models = train_without_cv_all(regression_models(), X_regr, y_regr)

    ## simple scaling of the data (center on zero, std)
    # print("Transformed:")
    # X_regr_scaled = StandardScaler().fit_transform(X_regr)


    """ Alpha Testing """
    # model_name = 'ridge'
    # print("Model:", model_name)
    # models[model_name] = alpha_testing(models[model_name], X_train, X_test, y_train, y_test)