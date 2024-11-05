import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# Load data
os.chdir('/work/home/dxdgroup02/lht/MOGONET/1008/mus')
RNA1 = pd.read_table("exp_bmi.txt", sep='\s+')

DNA1 = pd.read_table("snp_bmi.txt", sep='\s+')

RNA = RNA1.dropna(axis=1)
DNA = DNA1.dropna(axis=1)

y=RNA['BMI'].values

RNA = RNA.iloc[:,1:-4]
DNA = DNA.iloc[:,1:-4]

cols_with_negatives = RNA.columns[(RNA < 0).any()]

RNA = RNA.drop(columns=cols_with_negatives)

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.4)
selector.fit_transform(RNA)
RNA = RNA[RNA.columns[selector.get_support(indices=True)]]

selector.fit_transform(DNA)
DNA = DNA[DNA.columns[selector.get_support(indices=True)]]

RNA = RNA.astype(np.float64)
DNA = DNA.astype(np.float64)


from sklearn.preprocessing import MinMaxScaler


numeric_data = RNA.iloc[:,:].values


scaler = MinMaxScaler(feature_range=(0, 2))


scaled_data = scaler.fit_transform(numeric_data)


RNA.iloc[:,:] = scaled_data

# Combine RNA and DNA data
X1 = pd.concat([RNA, DNA], axis=1)
x = X1.values

