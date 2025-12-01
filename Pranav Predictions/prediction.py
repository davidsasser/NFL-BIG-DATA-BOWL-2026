import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import os

# get file path
current_directory = os.getcwd()
train_data_directory_path = current_directory + "/nfl-big-data-bowl-2026-analytics/114239_nfl_competition_files_published_analytics_final/train"
if os.path.isdir(train_data_directory_path):
    print(f"The directory '{train_data_directory_path}' exists.")
else:
    print(f"The directory '{train_data_directory_path}' does not exist.")

# Combine all input files
print("...Combining Input Files...")
input_files = sorted(glob.glob(os.path.join(train_data_directory_path, "input_2023_w*.csv")))
df_input = pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)
print("...Input Files Combined...")

# Combine all output files
print("...Combining Output Files...")
output_files = sorted(glob.glob(os.path.join(train_data_directory_path, "output_2023_w*.csv")))
df_output = pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)
print("...Output Files Combined...")


# Quick check
print(df_input.shape)
print(df_output.shape)

# Drop duplicates and missing values
print("...Dropping duplicates, replacing na values...")
df_input = df_input.drop_duplicates()
df_output = df_output.drop_duplicates()
# Basic numeric cleanup
df_input = df_input.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])

# Optional: Filter only players we predict
print("...filter by player_to_predict...")
df_input = df_input[df_input["player_to_predict"] == True]
print(df_input.head())

print("...managing dummies...")
#For gender, product_name and product_type create dummy variables
player_role_dum = pd.get_dummies(df_input["player_role"])
player_side_dum = pd.get_dummies(df_input["player_side"])
position_dum = pd.get_dummies(df_input["player_position"])
#Now concatenate the dum1 and dum2 with icps1
df_input = pd.concat([df_input,player_role_dum],axis=1)
df_input = pd.concat([df_input,player_side_dum],axis=1)
df_input = pd.concat([df_input,position_dum],axis=1)
#Drop one dummy from the gender and one from the produc_name to avoid collinearity
df_input.drop(['Defensive Coverage'], axis = 1, inplace=True)
df_input.drop(['Offense'], axis = 1, inplace=True)
df_input.drop(['MLB'], axis = 1, inplace=True)

# create age
print("...get ages...")
df_input["age"] = 2026 - pd.to_datetime(df_input["player_birth_date"]).dt.year
print("...input info after data edits...")
print(df_input.info())

# merge input and output
print("...merging input and output data...")
merged = df_input.merge(
    df_output[["game_id", "play_id", "nfl_id", "x", "y"]],
    on=["game_id", "play_id", "nfl_id"],
    suffixes=("_input", "_target")
)

# create merged set for model
print("...merge info...")
print(merged.info())

print("...creating X and y for model...")
X = merged[["absolute_yardline_number", "x_input", "y_input", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "Targeted Receiver", "Defense", "CB", "DE", "DT", "FB", "FS", "ILB", "LB", "NT", "OLB", "QB", "RB", "S", "SS", "T", "TE", "WR", "age"]]
y_x = merged["x_target"]
y_y = merged["y_target"]

# split train and test data
print("...splitting train and test data...")
X_train, X_test, yx_train, yx_test = train_test_split(X, y_x, test_size=0.2, random_state=42)
_, _, yy_train, yy_test = train_test_split(X, y_y, test_size=0.2, random_state=42)

# datetime
from datetime import datetime

print("...starting linear regression...")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr.fit(X_train, yx_train)
pred_x = lr.predict(X_test)
mse = mean_squared_error(yx_test, pred_x)
rmse = np.sqrt(mse)
print("Linear Regression RMSE: ", rmse)

print("...starting ridge regression...")
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, yx_train)
pred_x = ridge.predict(X_test)
mse = mean_squared_error(yx_test, pred_x)
rmse = np.sqrt(mse)
print("Pridge Regression RMSE: ", rmse)


# random forest
'''
print("...random forest regressor...")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
rf = RandomForestRegressor(n_estimators=100, random_state=42)
print("...random forest train...")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
rf.fit(X_train, yx_train)
print("...random forest predict...")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
pred_x = rf.predict(X_test)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("***RF X RMSE:", mean_squared_error(yx_test, pred_x, squared=False))
'''
print("...script complete...")