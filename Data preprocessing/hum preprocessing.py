import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load data
os.chdir('/work/home/dxdgroup02/lht/hum')
RNA = pd.read_table("gen_t_s2.txt")
DNA = pd.read_table("hot_mutation.txt")
PHE = pd.read_table("phe.txt")

RNA = RNA.drop(RNA.columns[[19264]], axis=1).iloc[:, 1:]
DNA = DNA.iloc[:, 1:]

# Variance threshold filtering
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.5)
RNA = RNA[RNA.columns[selector.fit(RNA).get_support(indices=True)]]

# Normalize RNA data
scaler = MinMaxScaler()
RNA.iloc[:, :] = scaler.fit_transform(RNA)

# Combine RNA and DNA data
X1 = pd.concat([RNA, DNA], axis=1)
x = X1.values
y = PHE['BMI'].values

