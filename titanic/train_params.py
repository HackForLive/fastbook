import random
from typing import Literal
import pandas as pd
import numpy as np
from pathlib import Path

random.seed(42)
# <-0.5,0.5>
shift = -0.5
rn = random.random() + shift
print(rn)


train_data = pd.read_csv(Path(__file__).parent / 'train.csv')
print(train_data.head(1))
print(train_data.columns)


for i in ['Sex', 'Age','Embarked', 'Parch']:
    print(f'Unique {i}:')
    print(train_data[i].unique())
# conditional variable - Survived
# Pclass_1 true/false, 'Pclass_2' true/false
# Male true/false
params = ['Pclass_1', 'Pclass_2', 'Male', 'Age_N', 'SibSp',
       'Parch', 'Fare_log_N', 'Embarked_S', 'Embarked_C']
cols = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']

# pd.notnull()

train_data_no_missing = train_data.dropna(axis=0)

def log_10(row):
    return np.log10(row + 1)

def equal_to_value(row, val) -> Literal[0, 1]:
    return (row == val).astype(int)

def normalise(row) -> float:
    m_val = max(row)
    return (row/m_val).astype(float)


# df_new = train_data_no_missing[
#     ['col1','col2','col3']
#     ].assign(
#     col1=lambda d: log_10(d['col1']), 
#     col2=lambda d: f2(d['col2']), 
#     col3=lambda d: f3(d['col3'])
# )

max_age = max(train_data_no_missing['Age'])
print(max_age)
df_new = train_data_no_missing.assign(
    Fare_log_N=lambda d: normalise(log_10(d['Fare'])), 
    Pclass_1=lambda d: equal_to_value(d['Pclass'], 1),
    Pclass_2=lambda d: equal_to_value(d['Pclass'], 2),
    Male=lambda d: equal_to_value(d['Sex'], 'male'),
    Embarked_S=lambda d: equal_to_value(d['Embarked'], 'S'),
    Embarked_C=lambda d: equal_to_value(d['Embarked'], 'C'),
    SibSp=lambda d: d['SibSp'],
    Age_N=lambda d: normalise(d['Age']),
    Parch=lambda d: normalise(d['Parch']),
)

y = df_new["Survived"]
X = df_new[params]



print(train_data_no_missing.head(20))
print(df_new.head(5))
print(y.head(5))
print(len(train_data))
print(len(df_new))


import torch

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(X_tensor.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(epoch, loss.item())