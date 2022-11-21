#!/usr/bin/env python
# coding: utf-8
# python train_CNNsq.py <kappa value> <sample_size>
# python train_CNNsq.py 0.15 100k

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from readTFR import *

os.environ['CUDA_VISIBLE_DEVICES']='2'

class CNNsq(tf.keras.Model):
    def __init__(self, name='CNNsq', dim_image=(75, 75, 2), n_class=3):
        super(CNNsq, self).__init__(name=name)
        
        """h2ptj Channel"""
        self.h2ptj = tf.keras.Sequential([
            # input tensor dimension must be (batch_size, rows, cols, channels)
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 0], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])

        """h2Qkj Channel"""
        self.h2Qkj = tf.keras.Sequential([
            # input tensor dimension must be (batch_size, rows, cols, channels)
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, 1], -1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (4,4), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(256, (6,6), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            tf.keras.layers.Dropout(0.5),
        ])
        
        """Output Layer"""
        self._output = tf.keras.layers.Dense(n_class, activation='softmax')
        
    @tf.function
    def call(self, inputs, training=False):
        """h2ptj"""
        latent_h2ptj = self.h2ptj(inputs)

        """h2Qkj"""
        latent_h2Qkj = self.h2Qkj(inputs)
        
        """Output"""
        latent_all = tf.concat([latent_h2ptj, latent_h2Qkj], axis=1)
        
        return self._output(latent_all)

def count_sample_size(y):
    # count sample size for 3 types
    size = [0,0,0]
    size[0] = (y == 0).sum()
    size[1] = (y == 1).sum()
    size[2] = (y == 2).sum()
    return size

def main():
    kappa = float(sys.argv[1])
    nevent = sys.argv[2]

    batch_size = 512
    dim_image = [[75, 75], [[-0.8, 0.8], [-0.8, 0.8]]]

    # Input datasets
    sample_dir = f'/home/r10222035/boosted_V_ML_test/sample/samples_kappa{kappa}-{nevent}/'
    print(f'Read data from {sample_dir}')
    data_tr = os.path.join(sample_dir, 'train.tfrecord')
    data_vl = os.path.join(sample_dir, 'valid.tfrecord')
    data_te = os.path.join(sample_dir, 'test.tfrecord')

    dataset_tr, tr_total = get_dataset(data_tr, repeat=False, 
                                    batch_size=batch_size, 
                                    dim_image=dim_image+[True], 
                                    shuffle=0)
    dataset_vl, vl_total = get_dataset(data_vl, repeat=False, 
                                    batch_size=batch_size, 
                                    dim_image=dim_image+[True], 
                                    shuffle=0)
    dataset_te, te_total = get_dataset(data_te, repeat=False, 
                                    batch_size=batch_size, 
                                    dim_image=dim_image+[True], 
                                    shuffle=0)

    labels_tr = np.vstack([x[1] for x in dataset_tr])
    labels_vl = np.vstack([x[1] for x in dataset_vl])
    labels_te = np.vstack([x[1] for x in dataset_te])

    y_tr = np.argmax(labels_tr, axis=1)
    y_vl = np.argmax(labels_vl, axis=1)
    y_te = np.argmax(labels_te, axis=1)

    train_size = count_sample_size(y_tr)
    print('Training size:', train_size)

    validation_size = count_sample_size(y_vl)
    print('Validation size:', validation_size)

    test_size = count_sample_size(y_te)
    print('Testing size:', test_size)

    # Training
    # Training parameters
    train_epochs = 500
    patience = 10
    min_delta = 0.
    learning_rate = 1e-4                                    
    save_model_name = f'best_model_CNNsq_kappa{kappa}-{nevent}/'

    # Create the model  
    history=0
    model = CNNsq(dim_image=[75,75,2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', 
                                                        verbose=1, save_best_only=True)

    history = model.fit(dataset_tr, validation_data=dataset_vl , epochs=train_epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])


    loaded_model = tf.keras.models.load_model(save_model_name)
    results = loaded_model.evaluate(dataset_te)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    # plot ROC
    labels = np.vstack([x[1] for x in dataset_te])
    predictions = loaded_model.predict(dataset_te).tolist()
    y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    fig, ax = plt.subplots(1,1, figsize=(6,5))

    particle_type = {0: '$W^+$', 1: '$W^-$', 2: '$Z$    '}
    AUC = [0,0,0]
    ACC = [0,0,0]
    for i in range(3):
        AUC[i] = roc_auc_score(y_test==i,  y_prob[:,i])
        fpr, tpr, threasholds = roc_curve(y_test==i, y_prob[:,i])
        
        # 計算最高的正確率
        accuracy_scores = []
        print('Calculating ACC')
        for threshold in tqdm(threasholds):
            accuracy_scores.append(accuracy_score(y_test==i,  y_prob[:,i]>threshold))

        accuracies = np.array(accuracy_scores)
        ACC[i] = accuracies.max() 
        
        ax.plot(fpr, tpr, label = f'{particle_type[i]} AUC = {AUC[i]:.3f}  ACC = {ACC[i]:.3f}')

    ax.set_title(f'ROC of CNN$^2$ $\kappa = {kappa}$')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    plt.savefig(f'figures/ROC_CNNsq_kappa{kappa}-{nevent}.png', facecolor='White', dpi=300)

    now = datetime.datetime.now()
    # Write training results
    file_name = 'CNNsq_training_results.csv'
    df = pd.DataFrame({'kappa': [kappa],
                    'Sample': [nevent],
                    'Train W+': [train_size[0]],
                    'Train W-': [train_size[1]],
                    'Train Z':  [train_size[2]],
                    'Validation W+': [validation_size[0]],
                    'Validation W-': [validation_size[1]],
                    'Validation Z':  [validation_size[2]],
                    'Test W+': [test_size[0]],
                    'Test W-': [test_size[1]],
                    'Test Z':  [test_size[2]],
                    'Overall ACC': [results[1]],
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
