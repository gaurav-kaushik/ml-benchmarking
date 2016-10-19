#!/usr/bin/env python

from __future__ import print_function
import argparse
from sys import exit
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn_functions import *
import matplotlib.pyplot as plt
import sklearn as sk
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-nf", "--noisy-features", action='store_true', help="Noisy features", required=False)
    parser.add_argument("-nn", "--noise-normal", action='store_true', help="Normal noise", required=False)
    parser.add_argument("-cv", "--cross-validation", type=int, default=5,
                        help="Cross-validation factor", required=False)
    parser.add_argument("-d", "--data", help="Data matrix (csv)", required=True)
    parser.add_argument("-t", "--target", help="Target matrix (csv)", required=True)
    args = vars(parser.parse_args())
    noise_normal = args['noise_normal']
    noisy_features = args['noisy_features']
    args_cv = args['cross_validation']
    data_file = args['data']
    target_file = args['target']

    # Set seed
    np.random.seed(0)

    """ Load Data """
    try:
        # Read target
        with open(target_file, "r") as t:
            df_sample = pd.read_csv(t, usecols=['sample_type'])

            # Convert categorical variable into indicators
            df_sample = pd.get_dummies(df_sample)
            label_indicators = df_sample.head() # store for your use
            df_target = df_sample.drop('sample_type_Solid Tissue Normal', axis=1) # get the Primary Tumor Samples as Labels

        # Read data
        with open(data_file) as d:
            df_data = pd.read_csv(d)
            df_data = df_data.drop('case', axis=1)
    except:
        print("Error in reading files")
        exit(1)

    # Store as numpy arrays
    X = df_data.values
    y = df_target.values.ravel() # use ravel to get appropriate dtype
    print("Shape (X, y):", X.shape, y.shape)

    """ Classification """

    # Add noise -- to add: add argparse for degree and rand_state
    if noise_normal:
        X = add_noise_normal(X, degree=1)
    if noisy_features:
        X = add_noisy_features(X, degree=200, rand_state=1)

    # Testing: linear model
    # model = sk.linear_model.LinearRegression()
    # model = train_without_cv(X, y, model)
    # print("Model trained.")

    # Testing: random forest model
    model = sk.ensemble.RandomForestClassifier()
    model = train_without_cv(X, y, model)
    print("Model trained.")

    plot_ROC(model, X, y, cv_fold=5, png_filename='test_class')
    print("ROC plots output")

    train_without_cv_all(X,y)
    print("Trained on all models w/o cv")


    """ Generate classifiers w/o cross-validation """
    print("Classifiers:")
    classifiers = train_without_cv_all(X, y)
    #
    """ Generate classifiers w/ cross-validation and ROC """
    for k, model in classifiers.items():
        plot_ROC(model, X, y, cv_fold=args_cv, png_filename=str(k))

    """ PCA & LDA that shizz """
    pca = PCA(n_components=2)
    pca3 = PCA(n_components=3)
    lda = LinearDiscriminantAnalysis(n_components=2)

    X_p2 = pca.fit(X).transform(X)
    X_p3 = pca3.fit(X).transform(X)
    X_l2 = lda.fit(X, y).transform(X)

    print("PCA: ", pca.explained_variance_ratio_)
    print("LDA: ", lda.explained_variance_ratio_)

    """ PCA Plot """
    # colors = ['blue', 'red', 'darkorange']
    # lw = 2
    # target_names = ["Primary Tumor", "Solid Tissue Normal"]
    # plt.figure()
    # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    #     plt.scatter(X_p2[y == i, 0], X_p2[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('PCA')
    # plt.show()

    """ LDA Plot (needs debugging) """
    # colors = ['blue', 'red', 'darkorange']
    # lw = 2
    # target_names = ["Primary Tumor", "Solid Tissue Normal"]
    # plt.figure()
    # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    #     plt.scatter(X_l2[y == i, 0], X_l2[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    # plt.title('LDA')
    # plt.show()

    """ 3D PCA Plot """
    fig = plt.figure(1, figsize=(4,3))
    plt.clf()
    ax = Axes3D(fig, rect=[0,0,0.95,1], elev=40, azim=120)
    plt.cla()
    targets = [('Primary Tumor', 0), ('Solid Tissue Normal', 1)]
    for name, label in targets:
        # ax.text3D(X_p3[y == label, 0].mean(),
        #           X_p3[y == label, 1].mean() + 1.5,
        #           name, horizontalalignment='center', bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        ax.scatter(X_p3[:,0], X_p3[:,1], X_p3[:,2], c=y, cmap=plt.cm.spectral)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()


    print("Analysis complete.")
