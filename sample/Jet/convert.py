#!/usr/bin/env python
# coding: utf-8
# python convert.py <npy_file_path>

import os
import sys

import numpy as np
from tqdm import tqdm
import itertools
from contextlib import ExitStack
import tensorflow as tf

def _list_float_feature(lst):
    """ Creates a feature object from a list of floats"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=lst))

def list_feature(labels):
    """ Create feature for list of features"""
    return _list_float_feature(labels)

# List of lists
def _list_of_lists_float_feature(list_of_lists):
    """ Creates a FeatureList object from a list of lists of floats"""
    list_of_features = [_list_float_feature(lst) for lst in list_of_lists]
    return tf.train.FeatureList(feature=list_of_features)

def image_feature(image):
    """ Create feature for images """
    return _list_of_lists_float_feature(image)

def get_sequence_example_object(data_element_dict):
    """ Creates a SequenceExample object from a dictionary for a single data element 
    data_element_dict is a dictionary for each element in .json file created by the fastjet code. 
    """
    # Context contains all scalar and list features
    context = tf.train.Features(
            feature=
            {
                'labels'  : list_feature(data_element_dict['labels']),
            }
    )
    
    # Feature_lists contains all lists of lists
    feature_lists = tf.train.FeatureLists(
            feature_list=
            {
                'pT'   : image_feature(data_element_dict['pT']),
                'Qk'   : image_feature(data_element_dict['Qk']),
            }
    )
                
    sequence_example = tf.train.SequenceExample(context       = context,
                                                feature_lists = feature_lists)
    
    return sequence_example

def determine_entry(entry, idx):

    if (entry[0]=='Z'):
        #print ("Z")
        entry[0]=[0,0,1]
        #N_Z += 1
    elif (entry[0]=='W+'):
        #print ("W+")
        entry[0]=[1,0,0]
        #N_Wp += 1
    elif (entry[0]=='W-'):
        #print ("W-")
        entry[0]=[0,1,0]
        #N_Wm += 1
    elif (entry[0]==[0,1,0]) or (entry[0]==[1,0,0]) or (entry==[0,0,1]):
        pass
    else:
        print ("not string entry, entry[0] is", entry[0], "at", idx)
        raise AssertionError
    return entry[0]

def create_TFRecord(npy_files):
    datasizes = []

    NameOfDirectory = os.path.split(npy_files[0])[0]

    if type(npy_files) == str:
        with open(npy_files.split('.npy')[0] + '.count') as f:
            datasizes.append(int(f.readline()))
        npy_files = [npy_files]
    elif type(npy_files) == list:
        for fname in npy_files:
            print (fname.split('.npy')[0] + '.count')
            with open(fname.split('.npy')[0] + '.count') as f:
                datasizes.append(int(f.readline()))
        dataset = np.array([np.load(npy_file, allow_pickle=True) for npy_file in npy_files])
           
    print ("datasizes in the npy", datasizes)
    
    datasize = sum(datasizes)
    trainsize = int(datasize*0.8)
    testsize  = int(datasize-trainsize)
    
    validsize = int(trainsize*0.2)
    trainsize = int(trainsize-validsize)

    idlist = np.array(list(itertools.chain.from_iterable([[idx]*datasizes[idx] for idx in range(len(datasizes))])), dtype=np.int64)
    np.random.shuffle(idlist)
    
    print(idlist)

    print ("Training, validation, and testing set are saved in "+NameOfDirectory)
    

    if not os.path.isdir(NameOfDirectory):
        os.mkdir(NameOfDirectory, 0o777)
    else:
        print ('directory already there.')
        #os.system('trash '+NameOfDirectory+'/*')
        os.system('ls '+NameOfDirectory)

    with tqdm(total=datasize) as pbar:
        with ExitStack() as stack:
            npy_readers = [stack.enter_context(open(npy_file, 'rb')) for npy_file in npy_files]

            tr_list = idlist[:trainsize]
            vl_list = idlist[trainsize:trainsize+validsize]
            te_list = idlist[trainsize+validsize:]

            N_Z, N_Wp, N_Wm = 0, 0, 0
            with tf.io.TFRecordWriter(NameOfDirectory +'/train.tfrecord') as tfwriter:
                for idx in tr_list:
                    entry = np.load(npy_readers[idx], allow_pickle=True)
                    entry[0] = determine_entry(entry, idx)
                    dict_obj = {'labels': entry[0], 'pT': entry[1], 'Qk': entry[2]}
                    sequence_example = get_sequence_example_object(dict_obj)
                    tfwriter.write(sequence_example.SerializeToString())
                    pbar.update(1)
            #print (N_Z, N_Wp, N_Wm)

            N_Z, N_Wp, N_Wm = 0, 0, 0
            with tf.io.TFRecordWriter(NameOfDirectory +'/valid.tfrecord') as tfwriter:
                for idx in vl_list:
                    entry = np.load(npy_readers[idx], allow_pickle=True)
                    entry[0] = determine_entry(entry, idx)
                    dict_obj = {'labels': entry[0], 'pT': entry[1], 'Qk': entry[2]}
                    sequence_example = get_sequence_example_object(dict_obj)
                    tfwriter.write(sequence_example.SerializeToString())
                    pbar.update(1)
            #print (N_Z, N_Wp, N_Wm)
            
            N_Z, N_Wp, N_Wm = 0, 0, 0
            with tf.io.TFRecordWriter(NameOfDirectory +'/test.tfrecord') as tfwriter:
                for idx in te_list:
                    entry = np.load(npy_readers[idx], allow_pickle=True)
                    entry[0] = determine_entry(entry, idx)
                    dict_obj = {'labels': entry[0], 'pT': entry[1], 'Qk': entry[2]}
                    sequence_example = get_sequence_example_object(dict_obj)
                    tfwriter.write(sequence_example.SerializeToString())
                    pbar.update(1)
            #print (N_Z, N_Wp, N_Wm)
 
    with open(NameOfDirectory +'/train.count', 'w+') as f:
        f.write('{0:d}\n'.format(trainsize))
    with open(NameOfDirectory +'/valid.count', 'w+') as f:
        f.write('{0:d}\n'.format(validsize))
    with open(NameOfDirectory +'/test.count', 'w+') as f:
        f.write('{0:d}\n'.format(testsize))

if __name__ == '__main__':
    filelist = sys.argv[1:]
    create_TFRecord(filelist)
