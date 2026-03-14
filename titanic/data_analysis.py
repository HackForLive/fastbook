import pandas as pd
from pathlib import Path
import logging

train_data = pd.read_csv(Path(__file__).parent / 'train.csv')


logging.info(train_data.describe(include='all'))
logging.info(train_data.info())

# key observation is not to drop all missing data, but to exclude Cabin -> or restrict
# 10  Cabin        204 non-null    object 

logging.info(train_data.duplicated().sum())