import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

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

def model_train(input_shape, filepath, model_save_path):
    input_shape = 100
    test_file1 = 'corpus.csv'
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    print(x.shape)
    print(y.shape)
    print(vocab_size)
    print(label_size)
    print(output_dictionary)
    print(inverse_word_dictionary)

if __name__ == '__main__':
    filepath = 'corpus.csv'
    input_shape = 180
    model_save_path = 'model.h5'
    model_train(input_shape, filepath, model_save_path)