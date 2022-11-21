#!/usr/bin/env python
# coding: utf-8
# python train_BDT.py <kappa value> <sample_size>
# python train_BDT.py 0.15 100k

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

def count_sample_size(y):
    # count sample size for 3 types
    size = [0,0,0]
    size[0] = (y == 0).sum()
    size[1] = (y == 1).sum()
    size[2] = (y == 2).sum()
    return size

def plot_distribution(X, y_true, y_predict, kappa, nevent):
    # plot mass charge distribution for 3 types
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].scatter(X[y_true==0][:,1], X[y_true==0][:,0], s=1, alpha=0.3, c='b', label='$W^+$')
    ax[0].scatter(X[y_true==1][:,1], X[y_true==1][:,0], s=1, alpha=0.3, c='g', label='$W^-$')
    ax[0].scatter(X[y_true==2][:,1], X[y_true==2][:,0], s=1, alpha=0.3, c='r', label='$Z$')

    ax[0].set_title(f'True $\kappa = {kappa}$')
    ax[0].set_xlabel('$\mathcal{Q}_\kappa$')
    ax[0].set_ylabel('$\mathcal{M}$ (GeV)')
    ax[0].set_ylim([40,140])
    ax[0].legend()

    ax[1].scatter(X[y_predict==0][:,1], X[y_predict==0][:,0], s=1, alpha=0.3, c='b', label='$W^+$')
    ax[1].scatter(X[y_predict==1][:,1], X[y_predict==1][:,0], s=1, alpha=0.3, c='g', label='$W^-$')
    ax[1].scatter(X[y_predict==2][:,1], X[y_predict==2][:,0], s=1, alpha=0.3, c='r', label='$Z$')

    ax[1].set_title(f'BDT prediction $\kappa = {kappa}$')
    ax[1].set_xlabel('$\mathcal{Q}_\kappa$')
    ax[1].set_ylabel('$\mathcal{M}$ (GeV)')

    ax[1].set_ylim([40,140])
    ax[1].legend()

    plt.savefig(f'figures/True_and_BDT_distribution_of_M_Qk_kappa{kappa}-{nevent}.png', facecolor='White', dpi=300)
    
def main():
    kappa = float(sys.argv[1])
    nevent = sys.argv[2]
    
    # Get training and testing sample
    sample_dir = f'/home/r10222035/boosted_V_ML_test/sample/samples_kappa{kappa}-{nevent}/'
    print(f'Read data from {sample_dir}')
    processes = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj']

    df_list = []
    for name in processes:
        sample_path = os.path.join(sample_dir, name + '_properties.txt')
        df = pd.read_csv(sample_path, index_col=0).replace('W+',0).replace('W-',1).replace('Z',2)
        df_list.append(df)

    df = pd.concat(df_list)
    X = np.array(df.drop('particle type', axis=1))
    y = np.array(df['particle type'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    train_size = count_sample_size(y_train)
    print('Training size:', train_size)

    test_size = count_sample_size(y_test)
    print('Testing size:', test_size)
    
    # Training GBDT
    GBDT = GradientBoostingClassifier()
    GBDT.fit(X_train, y_train)
    now = datetime.datetime.now()
    
    acc = GBDT.score(X_test, y_test)
    print(f'Overall ACC: {acc:.3}')
    
    # plot ROC
    GBDT_y_proba = GBDT.predict_proba(X_test)
    
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    AUC = [0,0,0]
    ACC = [0,0,0]
    particle_type = {0: '$W^+$', 1: '$W^-$', 2: '$Z$    '}
    for i in range(3):
        AUC[i] = roc_auc_score(y_test==i, GBDT_y_proba[:,i])

        Gbdt_fpr, Gbdt_tpr, Gbdt_threasholds = roc_curve(y_test==i, GBDT_y_proba[:,i])

        # 計算最高的正確率
        accuracy_scores = []
        print('Calculating ACC')
        for threshold in tqdm(Gbdt_threasholds):
            accuracy_scores.append(accuracy_score(y_test==i, GBDT_y_proba[:,i]>threshold))

        accuracies = np.array(accuracy_scores)
        ACC[i] = accuracies.max() 

        ax.plot(Gbdt_fpr, Gbdt_tpr, label = f'{particle_type[i]} AUC = {AUC[i]:.3f}  ACC = {ACC[i]:.3f}')

    ax.set_title(f'ROC of BDT $\kappa = {kappa}$')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    plt.savefig(f'figures/ROC_BDT_kappa{kappa}-{nevent}.png', facecolor='White', dpi=300)

    # plot distribution
    y_predict = GBDT.predict(X_test)
    plot_distribution(X_test, y_test, y_predict, kappa, nevent)

    # Write training results
    file_name = 'BDT_training_results.csv'
    df = pd.DataFrame({'kappa': [kappa],
                    'Sample': [nevent],
                    'Train W+': [train_size[0]],
                    'Train W-': [train_size[1]],
                    'Train Z':  [train_size[2]],
                    'Test W+': [test_size[0]],
                    'Test W-': [test_size[1]],
                    'Test Z':  [test_size[2]],
                    'Overall ACC': [acc],
                    'W+ AUC': [AUC[0]],
                    'W+ ACC': [ACC[0]],
                    'W- AUC': [AUC[1]],
                    'W- ACC': [ACC[1]],
                    'Z AUC':  [AUC[2]],
                    'Z ACC':  [ACC[2]],
                    'time': now,
                    })
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)

if __name__ == '__main__':
    main()