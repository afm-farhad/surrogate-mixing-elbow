import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/training_data_updated.csv")

X = df[['Hot_Inlet', 'Cold_Inlet']]
y = df['Outlet_std']
model = make_pipeline(StandardScaler(),
                      GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e-1)),
                                               normalize_y=True))
model.fit(X, y)

hot_space = np.linspace(0.4, 2, 50)
cold_space = np.linspace(0.2, 0.6, 50)
hot, cold = np.meshgrid(hot_space, cold_space)
grid_df = pd.DataFrame({"Hot_Inlet": hot.ravel(), "Cold_Inlet": cold.ravel()})

mean, std = model.predict(grid_df, return_std=True)
mean_map = mean.reshape(hot.shape)

plt.contourf(hot, cold, mean_map, levels=8, cmap="coolwarm")
plt.colorbar(label="Outlet temperature non-uniformity σ_T (K)")
plt.scatter(df["Hot_Inlet"], df["Cold_Inlet"], color="black", s=20, label="Fluent runs")
plt.xlabel("Hot inlet speed (m/s)")
plt.ylabel("Cold inlet speed (m/s)")
plt.title("Outlet temperature non uniformity from GP surrogate")
plt.legend()
plt.savefig("sigma_T/sigma_T_map.png", dpi=200)
plt.show()