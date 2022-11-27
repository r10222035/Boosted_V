#!/usr/bin/env python
# coding: utf-8
# python train_CNNsq.py <kappa value> <sample_size>
# python train_event_CNNsq.py 0.15 100k

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

os.environ['CUDA_VISIBLE_DEVICES']='0'

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

def count_sample_size(y, n_type=6):
    # count sample size 
    size = [(y == i).sum() for i in range(n_type)]
    return size

def main():
    kappa = float(sys.argv[1])
    nevent = sys.argv[2]

    batch_size = 512
    dim_image = [[75, 75], [[-3, 3], [-3, 3]]]

    # Input datasets
    sample_dir = f'/home/r10222035/Boosted_V/sample/event_samples_kappa{kappa}-{nevent}/'
    print(f'Read data from {sample_dir}')
    data_tr = os.path.join(sample_dir, 'train.tfrecord')
    data_vl = os.path.join(sample_dir, 'valid.tfrecord')
    data_te = os.path.join(sample_dir, 'test.tfrecord')

    dataset_tr, tr_total = get_dataset(data_tr, repeat=False, 
                                       batch_size=batch_size, 
                                       dim_image=dim_image+[True], 
                                       shuffle=0,
                                       N_labels=6
                                      )
    dataset_vl, vl_total = get_dataset(data_vl, repeat=False, 
                                       batch_size=batch_size, 
                                       dim_image=dim_image+[True], 
                                       shuffle=0,
                                       N_labels=6
                                      )
    dataset_te, te_total = get_dataset(data_te, repeat=False, 
                                       batch_size=batch_size, 
                                       dim_image=dim_image+[True], 
                                       shuffle=0,
                                       N_labels=6,
                                      )

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
    model_name = 'CNNsq'
    # Training parameters
    train_epochs = 500
    patience = 10
    min_delta = 0.
    learning_rate = 1e-4                                    
    save_model_name = f'best_model/best_model_event_CNNsq_kappa{kappa}-{nevent}/'

    # Create the model  
    history=0
    model = CNNsq(dim_image=[75,75,2], n_class=6)
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
    
    # Plot Loss Accuracy curve
    fig, ax = plt.subplots(1,1, figsize=(6,5))

    x = range(len(history.history['loss']))
    y_train = history.history['loss']
    y_validation = history.history['val_loss']

    ax.plot(x, y_train, label='Training')
    ax.plot(x, y_validation, label='Validation')

    ax.set_title('Loss across training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (categorical cross-entropy)')
    ax.legend()
    plt.savefig(f'figures/event_loss_curve_{model_name}_kappa{kappa}-{nevent}.png', facecolor='White', dpi=300, bbox_inches = 'tight')
#     plt.show()

    fig, ax = plt.subplots(1,1, figsize=(6,5))

    x = range(len(history.history['accuracy']))
    y_train = history.history['accuracy']
    y_validation = history.history['val_accuracy']

    ax.plot(x, y_train, label='Training')
    ax.plot(x, y_validation, label='Validation')

    ax.set_title('Accuracy across training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.savefig(f'figures/event_accuracy_curve_{model_name}_kappa{kappa}-{nevent}.png', facecolor='White', dpi=300, bbox_inches = 'tight')
#     plt.show()

    # plot ROC
    labels = np.vstack([x[1] for x in dataset_te])
    predictions = loaded_model.predict(dataset_te).tolist()
    y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)

    fig, ax = plt.subplots(1,1, figsize=(6,5))

    event_type = {0: '$W^+W^+$', 1: '$W^-W^-$', 2: '$ZZ$        ', 3:'$W^+Z$    ', 4:'$W^-Z$    ', 5:'$W^+W^-$'}
    AUC = np.zeros(6)
    ACC = np.zeros(6)
    for i in range(6):
        AUC[i] = roc_auc_score(y_test==i,  y_prob[:,i])
        fpr, tpr, thresholds = roc_curve(y_test==i, y_prob[:,i])

        # 計算最高的正確率
        accuracy_scores = []
        print('Calculating ACC')
        thresholds = np.array(thresholds)
        # 最多用 1000 個
        if len(thresholds) > 1000:
            thresholds = np.percentile(thresholds, np.linspace(0,100,1001))

        for threshold in tqdm(thresholds):
            accuracy_scores.append(accuracy_score(y_test==i,  y_prob[:,i]>threshold))

        accuracies = np.array(accuracy_scores)
        ACC[i] = accuracies.max() 

        ax.plot(fpr, tpr, label = f'{event_type[i]} AUC = {AUC[i]:.3f}  ACC = {ACC[i]:.3f}')

    ax.set_title(f'ROC of CNN$^2$ $\kappa = {kappa}$')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    plt.savefig(f'figures/event_ROC_{model_name}_kappa{kappa}-{nevent}.png', facecolor='White', dpi=300, bbox_inches = 'tight')

    now = datetime.datetime.now()
    # Write training results
    file_name = 'event_training_results.csv'
    event_type = ['W+W+', 'W-W-', 'ZZ', 'W+Z', 'W-Z', 'W+W-']
    data_dict = {'kappa': [kappa],
                'Sample': [nevent],
                'Model': [model_name],
                'Overall ACC': [results[1]],
                'time': [now],
                }

    for i, _type in enumerate(event_type):
        data_dict[f'Train {_type}'] = [train_size[i]]
        data_dict[f'Validation {_type}'] = [validation_size[i]]
        data_dict[f'Test {_type}'] = [test_size[i]]
        data_dict[f'{_type} AUC'] = [AUC[i]]
        data_dict[f'{_type} ACC'] = [ACC[i]]
    df = pd.DataFrame(data_dict)
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)

if __name__ == '__main__':
    main()