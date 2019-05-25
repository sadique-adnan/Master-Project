import numpy as np
import re
import os
import utils
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    #print('length of x_text',len(x_text))
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        #print('Label values:', label)

        labels.append(label)
    #print('labels',labels)
    y = np.array(labels)
    #print('x_test',x_test)
    #print('labels', y)
    return [x_text, y]


def get_datasets_tobacco(path='data/tobacco_full/'):
    """
    Loads Tobacco data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    arr = sorted(os.listdir(path))
    print('arr',arr)
    datasets = dict()
    class_value = 0
    datasets['data'] = []
    datasets['target'] = []
    datasets['target_names'] = []

    for input_file in arr:
        read_file = path + input_file
        data = list(open(read_file, "r").readlines())
        print('Data in each file',input_file,len(data))
        data = [s.strip() for s in data if len(s.strip())>0] # ignoring empty lines
        target = [class_value for x in data]
        datasets['data'].append(data)

        datasets['target'].append(target)
        datasets['target_names'].append(input_file)
        class_value = class_value + 1
        
   # print('The Data before flattening: ', datasets['data'])nvi
    datasets['data'] = utils.flatten_list(datasets['data'])
    datasets['target'] = utils.flatten_list(datasets['target'])
    #datasets['target_names'] = datasets['target_names']
    #print('The Data : ', datasets['data'])
    #print('The Target : ', datasets['target'])
    #print(len(datasets['data']))
    #print(len(datasets['target']))    
    #print('The Target Names: ', datasets['target_names'])

    return datasets


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    print('num_batches_per_epoch',num_batches_per_epoch)
    print('num_epochs',num_epochs)
    for epoch in range(num_epochs):
        print(epoch)
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
