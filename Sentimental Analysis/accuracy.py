from sklearn.metrics import accuracy_score
import pandas as pd
import sys

data1 = pd.read_csv('result.csv')
data2 = pd.read_csv('corpus.csv')

predict = data1['label']
real = data2['Rating']
result = 0
for num1, num2 in zip(predict, real):
    result += abs(num1 - num2) / 5
file = open("output.txt", "w")
old = sys.stdout
sys.stdout = file
print(1 - result / len(predict))
print(accuracy_score(predict, real))
file.close()
