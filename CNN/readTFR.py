import tensorflow as tf
import numpy as np

__all__ = ['parse_observ', 'parse_image', 'parse_flatten', 'parse_both', 'parse_efpout', 'get_dataset']

def parse_observ(data_record, dim_image, N_labels):
    """ Function to convert from TFRecord back to numbers, lists and arrays, and then to images."""
    # Single values and lists are stored in context_features
    context_features = {
        'labels' : tf.io.FixedLenFeature([N_labels], tf.float32),
        'obsers' : tf.io.FixedLenFeature([15], tf.float32),
    }
    
    sequence_features = None
    
    # Get single example
    context_data, sequence_data = tf.io.parse_single_sequence_example(data_record,
        context_features = context_features, 
        sequence_features = sequence_features)
    
    labels = tf.convert_to_tensor(context_data['labels'])
    obsers = tf.convert_to_tensor(context_data['obsers'])

    if dim_image[-1]:
        return obsers, labels
    else:
        return obsers

def parse_image(data_record, dim_image, N_labels):
    """ Function to convert from TFRecord back to numbers, lists and arrays, and then to images."""
    # Single values and lists are stored in context_features
    context_features = {
        'labels' : tf.io.FixedLenFeature([N_labels], tf.float32), #// number of input 
    }
    
    sequence_features = {
        'pT'  : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
        'Qk' : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
    }
    
    # Get single example
    context_data, sequence_data = tf.io.parse_single_sequence_example(data_record,
        context_features = context_features, 
        sequence_features = sequence_features)
    
    labels = tf.convert_to_tensor(context_data['labels'])
    pT  = tf.ensure_shape(tf.convert_to_tensor(sequence_data['pT']), list(dim_image[0]))
    Qk = tf.ensure_shape(tf.convert_to_tensor(sequence_data['Qk']), list(dim_image[0]))

    image = tf.stack([pT, Qk], axis=-1)

    if dim_image[-1]:
        return image, labels
    else:
        return image
    
def parse_flatten(data_record, dim_image, N_labels):
    """ Function to convert from TFRecord back to numbers, lists and arrays, and then to images."""
    # Single values and lists are stored in context_features
    context_features = {
        'labels' : tf.io.FixedLenFeature([N_labels], tf.float32),
    }
    
    sequence_features = {
        'pTj'  : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
        'Qkj' : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
    }
    
    # Get single example
    context_data, sequence_data = tf.io.parse_single_sequence_example(data_record,
        context_features = context_features, 
        sequence_features = sequence_features)
    
    labels = tf.convert_to_tensor(context_data['labels'])
    h2ptjl = tf.reshape(tf.ensure_shape(tf.convert_to_tensor(sequence_data['pTj']), list(dim_image[0])), [-1])

    image = h2ptjl

    if dim_image[-1]:
        return image, labels
    else:
        return image

def parse_both(data_record, dim_image, N_labels):
    """ Function to convert from TFRecord back to numbers, lists and arrays, and then to images."""
    # Single values and lists are stored in context_features
    context_features = {
        'labels' : tf.io.FixedLenFeature([N_labels], tf.float32),
        'obsers' : tf.io.FixedLenFeature([15], tf.float32),
    }
    
    sequence_features = {
        'h2ptl'  : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
        'h2ptj'  : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
        'h2ptjl' : tf.io.FixedLenSequenceFeature([dim_image[0][0]], tf.float32),
    }
    
    # Get single example
    context_data, sequence_data = tf.io.parse_single_sequence_example(data_record,
        context_features = context_features, 
        sequence_features = sequence_features)
    
    labels = tf.convert_to_tensor(context_data['labels'])
    indices = tf.range(15)
    obsers = tf.pad(tf.expand_dims(tf.convert_to_tensor(context_data['obsers']), axis=-1), tf.constant([[0, dim_image[0][0] - len(indices)],[0, 0]]), 'CONSTANT')
    h2ptl  = tf.ensure_shape(tf.convert_to_tensor(sequence_data['h2ptl']), list(dim_image[0]))
    h2ptj  = tf.ensure_shape(tf.convert_to_tensor(sequence_data['h2ptj']), list(dim_image[0]))
    h2ptjl = tf.ensure_shape(tf.convert_to_tensor(sequence_data['h2ptjl']), list(dim_image[0]))

    image = tf.concat([h2ptjl, obsers], axis=1)

    if dim_image[-1]:
        return image, labels
    else:
        return image

def parse_efpout(data_record, dim_image, N_labels):
    """ Function to convert from TFRecord back to numbers, lists and arrays, and then to images."""
    # Single values and lists are stored in context_features
    context_features = {
        'labels' : tf.io.FixedLenFeature([N_labels], tf.float32),
        'obsers' : tf.io.FixedLenFeature([15], tf.float32),
        'efpout' : tf.io.FixedLenFeature([102], tf.float32),
    }
    
    sequence_features = None
    
    # Get single example
    context_data, sequence_data = tf.io.parse_single_sequence_example(data_record,
        context_features = context_features, 
        sequence_features = sequence_features)
    
    labels = tf.convert_to_tensor(context_data['labels'])
    efpout = tf.convert_to_tensor(context_data['efpout'])
    obsers = tf.convert_to_tensor(context_data['obsers'])
    
    output = tf.concat([efpout, obsers], axis=0)

    if dim_image[-1]:
        return output, labels
    else:
        return output

def get_dataset(tfrecord_files, 
                batch_size=None, 
                repeat=True,
                shuffle=100,
                prefetch=1,
                dim_image=None,
                flatten=False,
                efpout=False,
                num_parallel_calls=None, N_labels=3):
    """Get tf.data.Dataset for a tfrecord file or list of tfrecord files. 
    repeat, shuffle, prefetch: settings for loading dataset
    """
    datasize = 0

    if type(tfrecord_files)==str:
        with open(tfrecord_files.split('.tfrecord')[0] + '.count') as f:
            datasize += int(f.readline())
        dataset = tf.data.TFRecordDataset([tfrecord_files])
    elif type(tfrecord_files)==list:
        for fname in tfrecord_files:
            with open(fname.split('.tfrecord')[0] + '.count') as f:
                datasize += int(f.readline())
        files = tf.data.Dataset.list_files(tfrecord_files, shuffle=False)
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=len(tfrecord_files))

    if dim_image[0] is not None:
        if len(dim_image) == 4:
            print ("The data set contains images and flatten data")
            dataset = dataset.map(lambda x: parse_both(x, dim_image, N_labels), num_parallel_calls=num_parallel_calls)
        elif flatten:
            print ("The data set contains flatten data")
            dataset = dataset.map(lambda x: parse_flatten(x, dim_image, N_labels), num_parallel_calls=num_parallel_calls)
        else:
            print ("The data set contains images")
            dataset = dataset.map(lambda x: parse_image(x, dim_image, N_labels), num_parallel_calls=num_parallel_calls)
    else:
        if efpout:
            print ("The data set contains efpout")
            dataset = dataset.map(lambda x: parse_efpout(x, dim_image, N_labels), num_parallel_calls=num_parallel_calls)
        else:
            print ("The data set contains observables")
            dataset = dataset.map(lambda x: parse_observ(x, dim_image, N_labels), num_parallel_calls=num_parallel_calls)

    if shuffle > 0:
        dataset = dataset.shuffle(buffer_size=shuffle)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    if repeat:
        dataset = dataset.repeat()

    if prefetch > 0:
        dataset = dataset.prefetch(prefetch)
    
    return dataset, datasize
