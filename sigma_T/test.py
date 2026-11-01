from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
df = pd.read_csv("data/training_data_updated.csv")
X = df[['Hot_Inlet', 'Cold_Inlet']]
y = df['Outlet_std']

model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e-1)), normalize_y=True))

model.fit(X,y)

test_df = pd.DataFrame({'Hot_Inlet' : [1.5, 0.8] , 'Cold_Inlet' : [0.32 , 0.50] })

mean, std = model.predict(test_df, return_std = True)
print(f"prediction : {mean}")
print(f"Uncertainity : {std}")
