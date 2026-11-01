import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/training_data_final.csv")
df["hot_share"] = (df["Outlet_Avg"] - 293.15) / 20
df["cold_share"] = 1 - df["hot_share"]
df["sigma_unmixed"] = 20 * np.sqrt(df["hot_share"] * df["cold_share"])
df["Mixing_Index"] = 1 - df["Outlet_std"] / df["sigma_unmixed"]
X = df[["Hot_Inlet", "Cold_Inlet"]]
y = df["Mixing_Index"]
model = make_pipeline(StandardScaler(),
                      GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e-1)),
                                               normalize_y=True))
model.fit(X, y)

hot_space = np.linspace(0.4, 2, 50)
cold_space = np.linspace(0.2, 0.6, 50)
hot, cold = np.meshgrid(hot_space, cold_space)
grid_df = pd.DataFrame({"Hot_Inlet": hot.ravel(), "Cold_Inlet": cold.ravel()})

mean, std = model.predict(grid_df, return_std=True)
std_map = std.reshape(hot.shape)

plt.contourf(hot, cold, std_map, levels=8,cmap="viridis")
plt.colorbar(label="Mixing_index_uncertainty_map")
plt.scatter(df["Hot_Inlet"], df["Cold_Inlet"], color="white", edgecolor="black", s=20, label="Fluent runs")
plt.xlabel("Hot inlet speed (m/s)")
plt.ylabel("Cold inlet speed (m/s)")
plt.title("Mixing index uncertainty map from GP surrogate")
plt.legend()
plt.savefig("Mixing_Index/mixing_index_uncertainty_map.png", dpi=200)
plt.show()