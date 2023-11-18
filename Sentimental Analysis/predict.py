import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics import accuracy_score

# 导入字典
with open('word_dict.pk', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('label_dict.pk', 'rb') as f:
    output_dictionary = pickle.load(f)

# 载入模型
model_save_path = 'model.h5'
lstm_model = load_model(model_save_path)
input_shape = 100

data = pd.read_csv('corpus.csv')
successSentence = []
failedSentence = []
result = []
possibility = []
datas = data['Review_Text'].copy()
# print(datas)
i = 0
for sentence in datas:
    # sentence = sentence.replace("\n","")
    print(i)
    i += 1
    try:
        # 数据预处理

        # sentence = "good"
        x = [[word_dictionary[word] for word in sentence]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)

        # 模型预测
        y_predict = lstm_model.predict(x)
        label_dict = {v:k for k,v in output_dictionary.items()}
        successSentence.append(sentence)
        possibility.append(y_predict[0][0])
        result.append(label_dict[np.argmax(y_predict)])

    except KeyError:
        failedSentence.append(sentence)

final = pd.DataFrame(successSentence)
final['possibility'] = possibility
final['label'] = result
final.rename(columns={0:'sentence'},inplace=True)
fail = pd.DataFrame(failedSentence)
fail.rename(columns={0:'sentence'},inplace=True)

final.to_csv('result.csv',index=False)
fail.to_csv('fail.csv',index=False)