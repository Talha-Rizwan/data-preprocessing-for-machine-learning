# Talha Rizwan Malik 19I-0652
# Syed Muhammad Ibtisam 19i-0422
# Muhammad Anser Qureshi 19I-0680
from collections import Counter
import pandas as pd
from scipy import spatial
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(first, second):
    cos_sim = dot(first, second) / (norm(first) * norm(second))
    return cos_sim


data = pd.read_csv('spam.csv')
first_c = data.columns[0]
data = data.drop([first_c], axis=1)  # deleting the first column
f1 = pd.Series(' '.join(data['Message']).lower().split()).value_counts()[:20]
data = pd.read_csv('bbc-text.csv')
first_c = data.columns[0]
data = data.drop([first_c], axis=1)  # deleting the first column
f2 = pd.Series(' '.join(data['text']).lower().split()).value_counts()[:20]
print('Cosine similarity : ', cosine_similarity(f1, f2))
