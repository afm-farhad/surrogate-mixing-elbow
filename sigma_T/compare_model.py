import pandas as pd
df = pd.read_csv("data/training_data_updated.csv")
X = df[['Hot_Inlet', 'Cold_Inlet']]
Y = df['Outlet_std']

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import LeaveOneOut,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
models = {
    'dummy' : DummyRegressor(),
    'linear_regressor' : LinearRegression(),
    'polynomial' : make_pipeline(PolynomialFeatures(degree=2),LinearRegression()),
    'gaussian' : make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level_bounds=(1e-10, 1e-1)) , normalize_y=True))
}

for name, model in models.items():
    prediction = cross_val_predict(model,X,Y, cv = LeaveOneOut())
    df['prediction'] = prediction
    df['error'] = (df['Outlet_std'] - df['prediction']).abs()
    print(f"{name}: MAE = {df['error'].mean():.3f}, R2 = {r2_score(df['Outlet_std'], df['prediction']):.3f}")
    print(f"predicted_value : {df['prediction'][4]} , True_value : {df['Outlet_std'][4]}")


