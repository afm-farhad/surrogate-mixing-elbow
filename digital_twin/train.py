import pandas as pd
import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('data/training_data_final.csv')
df['hot_share'] = (df['Outlet_Avg'] - 293.15)/20
df['cold_share'] = 1 - df['hot_share']
df['sigma_unmixed'] = 20 * np.sqrt(df['hot_share'] * df['cold_share'])
df['Mixing_Index'] = 1 - df['Outlet_std'] / df['sigma_unmixed']

X = df[['Hot_Inlet', 'Cold_Inlet']]
y = df['Mixing_Index']

model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e-1)), normalize_y=True))
model.fit(X,y)

joblib.dump(model, 'digital_twin/gp_model.joblib')
print('saved')