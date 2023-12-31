import tensorflow as tf
import sys

# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 导入数据
# 文件的数据中，特征为evaluation, 类别为label.
def load_data(filepath, input_shape=20):
    df = pd.read_csv(filepath, dtype={'Rating': str, 'Review_Text': str})

    # 标签及词汇表
    labels, vocabulary = list(df['Rating'].unique()), list(df['Review_Text'].unique())

    # 构造字符级别的特征
    string = ''
    num = 0
    for word in vocabulary:
        num += 1
        string += word

    vocabulary = set(string)

    # 字典列表
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    vocab_size = len(word_dictionary.keys())  # 词汇表大小
    print("词汇表大小")
    print(vocab_size)
    label_size = len(label_dictionary.keys())  # 标签类别数量
    print("标签类别数量")
    print(label_size)
    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df['Review_Text']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['Rating']]
    y = [to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary


# 创建深度学习模型， Embedding + LSTM + Softmax.

def create_LSTM(n_units, input_shape, output_dim, filepath):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape, mask_zero=True))
    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))  # n_units
    model.add(Dropout(0.2))  # 0.1 overfitting, 0.3 under-fitting
    model.add(Dense(label_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # adam can change into sgd

    # plot_model(model, to_file='./model_lstm.png', show_shapes=True)
    model.summary()

    return model


def load(filepath, input_shape=20):
    df = pd.read_csv(filepath, dtype={'Rating': str, 'Review_Text': str})

    # 标签及词汇表
    labels, vocabulary = list(df['Rating'].unique()), list(df['Review_Text'].unique())

    # 构造字符级别的特征
    string = ''
    num = 0
    for word in vocabulary:
        num += 1
        string += word

    vocabulary = set(string)

    # 字典列表
    word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}

    label_size = len(label_dictionary.keys())  # 标签类别数量
    # 序列填充，按input_shape填充，长度不足的按0补充
    x = [[word_dictionary[word] for word in sent] for sent in df['Review_Text']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['Rating']]
    y = [to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, label_size


def predict(test_x, test_y, lstm_model, output_dictionary):
    N = test_x.shape[0]  # 测试的条数
    print(N)
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N + 1, 1)):
        # sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        label_true = output_dictionary[np.argmax(test_y[start:end])]
        # print(''.join(sentence), label_true, label_predict) # 输出预测结果
        predict.append(label_predict)
        label.append(label_true)

    acc = accuracy_score(predict, label)  # 预测准确率
    print('模型在测试集上的准确率为: %s.' % acc)


# 模型训练
def model_train(input_shape, filepath, model_save_path):
    input_shape = 100
    test_file1 = 'corpus.csv'
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    test_x1, test_y1, c = load(test_file1, input_shape)

    # 模型输入参数，需要自己根据需要调整
    n_units = 100  # 网络的单元数
    batch_size = 32  # 每轮训练的数据量的大小
    epochs = 20  # 训练轮数
    output_dim = 20

    # 模型训练
    lstm_model = create_LSTM(n_units, input_shape, output_dim, filepath)
    lstm_model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    lstm_model.save(model_save_path)

    predict(test_x1, test_y1, lstm_model, output_dictionary)

if __name__ == '__main__':
    filepath = 'corpus.csv'
    input_shape = 180
    model_save_path = 'model.h5'
    model_train(input_shape, filepath, model_save_path)