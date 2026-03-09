import random
from typing import Literal
import pandas as pd
import numpy as np
from pathlib import Path
import torch

test_data = pd.read_csv(Path(__file__).parent / 'test.csv')
# train_data = pd.read_csv(Path(__file__).parent / 'train.csv')

# Columns
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
params = ['Pclass_1', 'Pclass_2', 'Male', 'Age_N', 'SibSp',
       'Parch', 'Fare_log_N', 'Embarked_S', 'Embarked_C']

def log_10(row):
    return np.log10(row + 1)

def equal_to_value(row, val) -> Literal[0, 1]:
    return (row == val).astype(int)

def normalise(row) -> float:
    m_val = max(row)
    return (row/m_val).astype(float)

df_new = test_data.assign(
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

p_id = df_new["PassengerId"]
# y = df_new["Survived"]
X = df_new[params]

X_test_tensor = torch.tensor(X.values, dtype=torch.float32)

# view(-1, 1) reshapes a tensor into a 2‑D column vector with one column 
# and as many rows as needed. -1 tells PyTorch to infer that dimension size.

#y_test_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(X_test_tensor.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

model.load_state_dict(torch.load(Path(__file__).parent / "model.pth"))

with torch.no_grad():
    logits = model(X_test_tensor)
    preds = torch.sigmoid(logits)

pred_labels = (preds > 0.5).squeeze(-1).to(torch.int64).tolist()

# accuracy = (pred_labels == y_test_tensor).float().mean()

# print("Accuracy:", accuracy.item())

res = pd.DataFrame(columns=['PassengerId','Survived'], data=zip(list(p_id) , pred_labels))
res.to_csv(Path(__file__).parent / 'result.csv', encoding='utf-8', index=False)