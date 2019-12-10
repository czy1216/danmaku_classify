import numpy as np
np.random.seed(0)

from keras.models import Model
from keras.layers import Dense, Input, LSTM, Activation, Bidirectional, Dot, Concatenate, Dropout
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(1)
from keras.initializers import glorot_uniform
from danmu_utils import read_dataset_intent_all, div_train_and_test, get_embedding, softmax, sentence_to_indices

# danmu_semantic

file_path = 'data/danmu_seg.txt'
voc_embedding_path = 'tools/voc_embedding.json'

# 取出数据,标注只取情感
m, Tx, x, y_sentiment = read_dataset_intent_all(file_path)
# 取出词向量表
voc_embedding, word_to_index, index_to_word = get_embedding(voc_embedding_path)

#预训练Embedding层。每个词有一个编号，一个句子转换成一个编号序列，embedding层将每个编号转换为向量(以权重形式)，组成张量输入。
def pretrained_embedding_layer(voc_embedding, word_to_index):
    emb_dim = len(list(voc_embedding.values())[1])
    voc_len = len(word_to_index) + 1

    emb_matrix = np.zeros((voc_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = np.array(voc_embedding[word])

    embedding_layer =  Embedding(voc_len, emb_dim, trainable = False)
    embedding_layer.build((None,))  
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

#构建模型
def sentiment_model(Tx, input_shape, voc_embedding, word_to_index):
    #序列长度为T，控制one_step_attention()的循环次数
    sentence_indices = Input(shape=input_shape, dtype='int64')

    #embedding层
    embedding_layer = pretrained_embedding_layer(voc_embedding, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    #创建基本模型通过X连接各层
    X = Bidirectional(LSTM(256, return_sequences = True), merge_mode='sum')(embeddings)
    X = LSTM(128, return_sequences = False)(X)
    X = Dropout(0.5)(X)
    X = Dense(6, activation = 'softmax')(X)
    X = Activation('softmax')(X)

    model = Model(inputs = sentence_indices, outputs = X)
    return model

#计算词汇表中零向量个数
def count_zero_vector(voc_embedding):
    emb_dim = len(list(voc_embedding.values())[1])

    i = 0
    for j in voc_embedding.values():
        if (j == np.zeros((emb_dim,))).all():
            i = i + 1
    per = i/len(voc_embedding)
    return i, per

def train(m, Tx, x, y_sentiment, voc_embedding, word_to_index):
    model = sentiment_model(Tx, (Tx,), voc_embedding, word_to_index)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #将弹幕列表，转化为sentence-indices二维数组
    x_indices = sentence_to_indices(x, word_to_index, m, Tx)
    #取测试集与训练集
    x_train_indices, x_test_indices, y_train_oh, y_test_oh = div_train_and_test(x_indices, y_sentiment, num_classes=6, propertion=0.8)
    model.fit(x_train_indices, y_train_oh, epochs=50, shuffle=True, batch_size=64)
    #evaluate
    loss, acc = model.evaluate(x_test_indices, y_test_oh)
    print("Loss = ", loss)
    print("Test accuracy = ", acc)
    
    '''
    #保存模型与参数
    model_json = model.to_json()
    with open ('model/sentiment/model_noatt_three_60.json', 'w') as json_f:
        json_f.write(model_json)
    model.save_weights('model/sentiment/model_weights_noatt_three_60.h5')
    '''

    '''
    ## 用于二分类
    num_train = len(x_train_indices)
    y_sentiment_test = y_sentiment[num_train:]
    pred = model.predict(x_test_indices)
    #输出弹幕极性
    print('弹幕极性(数值越大，情感越倾向于正面)：')
    for i in range(m-num_train-1):
        sentence = ''
        for ind in list(x_test_indices[int(i),:]):
            if ind != 0:
                sentence = sentence + index_to_word[int(ind)]
            else:
                continue
        print(sentence + ':' + str(pred[i][1]))
    #输出错误结果
    for i in range(m-num_train-1):
        sentence = ''
        num = np.argmax(pred[i])
        if(num != y_sentiment_test[i]):
            for ind in list(x_test_indices[int(i),:]):
                if ind != 0:
                    sentence = sentence + index_to_word[int(ind)]
                else:
                    continue
            print('Expected label:'+ str(y_sentiment_test[i]) + ' prediction: ' + sentence + ' ' + str(num))
    '''

if __name__ == "__main__":
    train(m, Tx, x, y_sentiment, voc_embedding, word_to_index)