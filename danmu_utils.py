import numpy as np 
import json
import jieba
from bert_serving.client import BertClient
import keras.backend as K
from keras.models import model_from_json, Input, Model
import pandas as pd
import tensorflow as tf
import h5py

#读取数据集，获取情感信息(情感3分类)
def read_dataset_sentiment_three(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()

        # 随机化数据集中数据的顺序
        np.random.seed(10)
        data_index = np.random.permutation(len(data))
        data = np.array(data)[data_index]
        data = list(data)
        num_data = len(data)

        # 数据x为列表，标注y为numpy数组
        y_sentiment = np.zeros((num_data,),dtype=int)
        x = []
        len_series = pd.DataFrame(np.zeros((num_data,), dtype=int))
        for i in range(num_data):
            line = data[i].strip().split('|')
            danmu_list = line[:-2]
            len_series[0][i] = len(danmu_list)

            x.append(danmu_list)
            y_sentiment[i] = int(line[-2])

        # 用len_list得到弹幕长度的0.99分位数，即100条弹幕最多有5条信息不全
        Tx = int(len_series.quantile(0.95)[0]) + 1

    return num_data, Tx, x, y_sentiment

#读取数据集，获取相关信息(只读情感，用于二分类)
def read_dataset_sentiment(filepath):
    with open(filepath, 'r') as f:
        data_has_2 = f.readlines()
        data_unshuffle = []
        
        # 清除情感标记为2的弹幕
        for line in data_has_2:
            if int(line.strip().split('|')[-2]) != 2:
                data_unshuffle.append(line)

        # 随机化数据集中数据的顺序
        np.random.seed(10)
        data_index = np.random.permutation(len(data_unshuffle))
        data = np.array(data_unshuffle)[data_index]
        data = list(data)
        num_data = len(data)

        # 数据x为列表，标注y为numpy数组
        y_sentiment = np.zeros((num_data,),dtype=int)
        x = []
        len_series = pd.DataFrame(np.zeros((num_data,), dtype=int))
        for i in range(num_data):
            line = data[i].strip().split('|')
            danmu_list = line[:-2]
            len_series[0][i] = len(danmu_list)

            x.append(danmu_list)
            y_sentiment[i] = int(line[-2])

        # 用len_list得到弹幕长度的0.99分位数，即100条弹幕最多有5条信息不全
        Tx = int(len_series.quantile(0.95)[0]) + 1

    return num_data, Tx, x, y_sentiment

#读取数据集，获取相关信息(只读句子类型)
def read_dataset_intent(filepath):
    with open(filepath, 'r') as f:
        unshuffle_data = f.readlines()
        
        # 随机化数据集中数据的顺序
        np.random.seed(10)
        data_index = np.random.permutation(len(unshuffle_data))
        data = np.array(unshuffle_data)[data_index]
        data = list(data)
        num_data = len(data)

        # 数据x为列表，标注y为numpy数组
        y_intent = np.zeros((num_data,),dtype=int)
        x = []
        len_series = pd.DataFrame(np.zeros((num_data,), dtype=int))
        for i in range(num_data):
            line = data[i].strip().split('|')
            danmu_list = line[:-2]
            len_series[0][i] = len(danmu_list)

            x.append(danmu_list)
            y_intent[i] = int(line[-1])
            if y_intent[i]>1:
                print(line)

        # 用len_list得到弹幕长度的0.95分位数，即100条弹幕最多有5条信息不全
        Tx = int(len_series.quantile(0.95)[0]) + 1

    return num_data, Tx, x, y_intent

#读取数据集，获取相关信息(text-cnn六分类)
def read_dataset_intent_all(filepath):
    with open(filepath, 'r') as f:
        unshuffle_data = f.readlines()
        
        # 随机化数据集中数据的顺序
        np.random.seed(10)
        data_index = np.random.permutation(len(unshuffle_data))
        data = np.array(unshuffle_data)[data_index]
        data = list(data)
        num_data = len(data)
        
        # 数据x为列表，标注y为numpy数组
        y_intent = np.zeros((num_data,),dtype=int)
        x = []
        len_series = pd.DataFrame(np.zeros((num_data,), dtype=int))
        for i in range(num_data):
            line = data[i].strip().split('|')
            danmu_list = line[:-2]
            len_series[0][i] = len(danmu_list)

            x.append(danmu_list)
            y_intent[i] = int(line[-2])*2 + int(line[-1])

        # 用len_list得到弹幕长度的0.95分位数，即100条弹幕最多有5条信息不全
        # Tx = int(len_series.quantile(0.95)[0]) + 1
        Tx = 16
    return num_data, Tx, x, y_intent

#读取数据集，生成bert句子向量，用于bert分类
def read_dataset_bert(filepath):
    with open(filepath, 'r') as f:
        bc = BertClient()
        unshuffle_data = f.readlines()
        
        # 随机化数据集中数据的顺序
        np.random.seed(10)
        data_index = np.random.permutation(len(unshuffle_data))
        data = np.array(unshuffle_data)[data_index]
        data = list(data)
        num_data = len(data)
        
        # 数据x为列表，标注y为numpy数组
        y_intent = np.zeros((num_data,),dtype=int)
        x = np.zeros((num_data, 768), dtype='float32')

        for i in range(num_data):
            line = data[i].strip().split('|')
            y_intent[i] = int(line[-2])*2 + int(line[-1])

            s = ''
            for w in line[:-2]:
                s += w 
            
            x[i] = bc.encode([s]).reshape(768)            

    return num_data, x, y_intent

# 读取数据生成所有迭代batch(text-cnn)
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
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

# 读取数据生成所有迭代batch(bert分类)
def batch_iter_bert(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
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

# 读取词向量文件，获取词向量词典
def get_embedding(filepath):
    with open(filepath, 'r') as f:
        voc_embedding = json.load(f)

    #生成词与索引的字典
    word_to_index = {}
    index_to_word = {}
    i = 1
    for word in sorted(list(voc_embedding.keys())):
        index_to_word[i] = word
        word_to_index[word] = i
        i = i + 1

    return voc_embedding, word_to_index, index_to_word

# 取词库向量转化为张量
def get_index_to_embedding(voc_embedding, index_to_word):
    voc_size = len(voc_embedding)
    embedding_size = len(voc_embedding[index_to_word[1]])
    index_to_embedding = np.zeros((voc_size+1, embedding_size), dtype='float32')
    for i in range(1, voc_size+1):
        index_to_embedding[i] = np.array(voc_embedding[index_to_word[i]])
    return index_to_embedding

#将弹幕转化为sentence-indices二维数组
def sentence_to_indices(x, word_to_index, m, Tx):
    x_indices = np.zeros((m, Tx))
    for i in range(m):
        if Tx <= len(x[i]): 
            for j in range(Tx):
                x_indices[i][j] = word_to_index[x[i][j]]
        else:
            j = 0
            for word in x[i]:
                x_indices[i][j] = word_to_index[word]
                j = j+1

    return x_indices

#数字标注转化为热独码
def convert_to_one_hot(y, num_classes):
    Y = np.eye(num_classes)[y.reshape(-1)]
    return Y

#多维softmax
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

#分割数据集为测试集与验证集(bert)
def div_train_and_test_bert(x, y, num_classes, propertion):
    num_train = int(len(x)*propertion)
    x_train = x[:num_train]
    x_test = x[num_train:]
    y_oh = convert_to_one_hot(y, num_classes)
    y_train_oh = y_oh[:num_train, :]
    y_test_oh = y_oh[num_train:, :]
    return  x_train, x_test, y_train_oh, y_test_oh

#分割数据集为测试集与验证集
def div_train_and_test(x_indices, y, num_classes, propertion):
    num_train = int(len(x_indices)*propertion)
    x_train_indices = x_indices[:num_train, :]
    x_test_indices = x_indices[num_train:, :]
    y_oh = convert_to_one_hot(y, num_classes)
    y_train_oh = y_oh[:num_train, :]
    y_test_oh = y_oh[num_train:, :]
    return  x_train_indices, x_test_indices, y_train_oh, y_test_oh

#通过情感模型获取情感向量（词粒度）
def sentiment_vec(x_indices):
    with open('model/sentiment/model_noatt.json','r') as json_f:
        loaded_model_json = json_f.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model/sentiment/model_weights_noatt.h5')

    input_layer_name = 'input_1'
    output_layer_name = 'lstm_2'

    intermediate_layer_model = Model(inputs=loaded_model.get_layer(input_layer_name).input, outputs=loaded_model.get_layer(output_layer_name).output)
    sentiment_embedding = intermediate_layer_model.predict(x_indices)

    return sentiment_embedding

#通过bert模型获取bert句向量
def bert_vec(x_indices, index_to_word):
    bc = BertClient()
    x_indices = x_indices.astype('int')
    bert_embedding = np.zeros((x_indices.shape[0], 768))
    for i in range(x_indices.shape[0]):
        s = ''
        for j in list(x_indices[i]):
            if j == 0:
                s += ''
            else:
                s += index_to_word[j]
        
        bert_embedding[i] = bc.encode([s]).reshape(768)           

    return bert_embedding

# 读取数据生成所有迭代batch(包括情感向量与bert向量)
def batch_iter_with_sentiment_bert(data, batch_size, num_epochs, index_to_word, shuffle=True):
    data = np.array(data)
    x_indices = np.array(list(zip(*data))[0])
    # 生成情感向量
    sentiment_embedding = sentiment_vec(x_indices)
    # 生成bert向量
    bert_embedding = bert_vec(x_indices, index_to_word)

    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_sentiment_embedding = sentiment_embedding[shuffle_indices]
            shuffled_bert_embedding = bert_embedding[shuffle_indices]
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_sentiment_embedding = sentiment_embedding
            shuffled_bert_embedding = bert_embedding
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index], shuffled_sentiment_embedding[start_index:end_index], shuffled_bert_embedding[start_index:end_index]

# 读取数据生成所有迭代batch(包括情感向量)
def batch_iter_with_sentiment(data, batch_size, num_epochs, shuffle=True):
    # 生成情感向量
    #sentiment_embedding = np.load('tools/sentiment_sentence_emb.npy')
    data = np.array(data)
    x_indices = np.array(list(zip(*data))[0])
    sentiment_embedding = sentiment_vec(x_indices)

    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_sentiment_embedding = sentiment_embedding[shuffle_indices]
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
            shuffled_sentiment_embedding = sentiment_embedding
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index], shuffled_sentiment_embedding[start_index:end_index]