from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
df = pd.read_csv("data/training_data_updated.csv")
df["hot_share"] = (df["Outlet_Avg"] - 293.15) / 20
df["cold_share"] = 1 - df["hot_share"]
df["sigma_unmixed"] = 20 * np.sqrt(df["hot_share"] * df["cold_share"])
df["Mixing_Index"] = 1 - df["Outlet_std"] / df["sigma_unmixed"]
X = df[["Hot_Inlet", "Cold_Inlet"]]
y = df["Mixing_Index"]

model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e-1)), normalize_y=True))

model.fit(X,y)

test_df = pd.DataFrame({'Hot_Inlet' : [1.5, 0.8] , 'Cold_Inlet' : [0.32 , 0.50] })

mean, std = model.predict(test_df, return_std = True)
print(f"prediction : {mean}")
print(f"Uncertainity : {std}")
