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

train_data = train_data.drop(columns=['Cabin'])
train_data_no_missing = train_data.dropna(axis=0)

print(f"{len(train_data) =}")
print(f"{len(train_data_no_missing) =}")

def log_10(row):
    return np.log10(row + 1)

def equal_to_value(row, val) -> Literal[0, 1]:
    return (row == val).astype(int)

def normalise(row) -> float:
    m_val = max(row)
    return (row/m_val).astype(float)

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



import torch

X_tensor = torch.tensor(X.values, dtype=torch.float32)

# view(-1, 1) reshapes a tensor into a 2‑D column vector with one column 
# and as many rows as needed. -1 tells PyTorch to infer that dimension size.
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

import torch.nn as nn


# X_tensor.shape[1] = N → number of input features.

# Input → Dense layer
#
# Takes N input features
# Produces 16 neurons
# y = W*x + b
# W → weight matrix (16 × N)
# b → bias (16)
# nn.Linear(X_tensor.shape[1], 16)

model = nn.Sequential(
    nn.Linear(X_tensor.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# Binary Cross Entropy loss:

# Loss=−[ylog⁡(p)+(1−y)log⁡(1−p)]
# y = true label
# p = predicted probability
# criterion = nn.BCELoss()
# more stable then above
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30000):
    pred = model(X_tensor)
    # error
    loss = criterion(pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(epoch, loss.item())


# check final params
# for name, param in model.named_parameters():
#     print(name, param.data)

torch.save(model.state_dict(), Path(__file__).parent / "model.pth")
