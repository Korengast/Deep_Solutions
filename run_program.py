__author__ = "Koren Gast"


import sys
sys.path.append("../src")
import pandas as pd
import pickle
from utils import solve_eq_string, is_number, get_vocabulary, get_clean_varsAndEqn, get_max
from text_to_template import number_parsing
from models.models import DiltonModel
from models.models import EncoderDecoder_model
import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
pd.set_option('max_colwidth', 800)

train_data = load_data.load_alldata()
dev_data = pd.read_json("data/dev_data.json")
test_data = pd.read_json("data/test_data.json")

train_dev = get_clean_varsAndEqn(train_data.append(dev_data, sort=False))
txt_vocab = np.array([list(get_vocabulary(train_dev['text']))]).reshape(-1,1)
var_vocab = np.array([list(get_vocabulary(train_dev['clean_vars'].apply(lambda l: ", ".join(l))))]).reshape(-1,1)
eqn_vocab = np.array([list(var_vocab[0])+['+', '-', '*', '/', '^', '(', ')', '=']]).reshape(-1,1)

txt_lbl_encoder = LabelEncoder()
var_lbl_encoder = LabelEncoder()
eqn_lbl_encoder = LabelEncoder()

txt_lbl = txt_lbl_encoder.fit_transform(txt_vocab).reshape(-1,1)
var_lbl = var_lbl_encoder.fit_transform(var_vocab).reshape(-1,1)
eqn_lbl = eqn_lbl_encoder.fit_transform(eqn_vocab).reshape(-1,1)

txt_oh_encoder = OneHotEncoder()
var_oh_encoder = OneHotEncoder()
eqn_oh_encoder = OneHotEncoder()

txt_oneHot = txt_oh_encoder.fit_transform(txt_lbl)
var_oneHot = var_oh_encoder.fit_transform(var_lbl)
eqn_oneHot = eqn_oh_encoder.fit_transform(eqn_lbl)

max_vars = get_max(train_dev, 'clean_vars')
max_eqn = get_max(train_dev, 'clean_eqn')

input_shape = ...
output_shape = ...
model = EncoderDecoder_model(input_shape=input_shape, output_shape=output_shape)
model.fit(math_train)

## print evaluation result
print(f'result score on train: {model.score(math_train,frac=0.1,verbose=False,use_ans=True)}')
print(f'result score on test: {model.score(math_test,frac=1,verbose=True,use_ans=True)}')