import os
import numpy as np
import pandas as pd


# Load data
os.chdir('/work/home/dxdgroup02/lht/pig')
RNA = pd.read_table("gen_t.txt")
DNA = pd.read_table("SNP_t.txt")
PHE = pd.read_table("phe.txt")

RNA = RNA.iloc[:, 1:]
DNA = DNA.iloc[:, 1:]

# Feature selection
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.5)
RNA = RNA[RNA.columns[selector.fit(RNA).get_support(indices=True)]]
DNA = DNA[DNA.columns[selector.fit(DNA).get_support(indices=True)]]

# Convert to float before normalization to avoid FutureWarning
RNA = RNA.astype(np.float64)
DNA = DNA.astype(np.float64)

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
RNA.iloc[:, :] = scaler.fit_transform(RNA)
DNA.iloc[:, :] = scaler.fit_transform(DNA)

# Combine RNA and DNA data
X1 = pd.concat([RNA, DNA], axis=1)
x = X1.values
y = PHE['IMF'].values
