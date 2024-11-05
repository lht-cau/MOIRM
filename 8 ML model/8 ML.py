
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.base import BaseEstimator, RegressorMixin
import logging
import random


from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score


####1.Divide the training set and testing set according to the appropriate proportion based on the dataset structure###
####2.Set hyperparameters for grid search and cross validation###
####3.Train the optimal model using the entire dataset based on the best hyperparameters###
####4.Ranking feature importance based on the best trained model using feature permutation method####


####XGBRegressor####

xgb_model = XGBRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7]
}


grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)


print("XGB Starting Grid Search...")
grid_search.fit(x, y)


print("Best parameters found: ", grid_search.best_params_)


best_xgb_model = XGBRegressor(**grid_search.best_params_)


print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_xgb_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_xgb_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)


avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)


print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)


print("Training the best model...")
best_xgb_model.fit(x, y)


print("Evaluating the best model on all set...")
y_pred = best_xgb_model.predict(x)


test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)


print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_xgb_model.pkl'
joblib.dump(best_xgb_model, model_path)
print(f"XGB Best model saved to {model_path}")


from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.metrics import mean_absolute_error as mae

from IPython.display import display


REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_xgb_model = joblib.load('best_xgb_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_xgb_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_xgb_importance.csv', index=False)




####LinearRegression####

from sklearn.linear_model import LinearRegression

LR_model = LinearRegression()


param_grid = {
    'copy_X': [True, False],
    'fit_intercept': [True, False]
}


grid_search = GridSearchCV(estimator=LR_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)


print("LR Starting Grid Search...")
grid_search.fit(x, y)


print("Best parameters found: ", grid_search.best_params_)


best_LR_model = LinearRegression(**grid_search.best_params_)


print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []
for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_LR_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_LR_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)


avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)


print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)


print("Training the best model...")
best_LR_model.fit(x,y)


print("Evaluating the best model on all set...")
y_pred = best_LR_model.predict(x)


test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)


print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_LR_model.pkl'
joblib.dump(best_LR_model, model_path)
print(f"LR Best model saved to {model_path}")



REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_LR_model = joblib.load('best_LR_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_LR_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_LR_importance.csv', index=False)






####DecisionTreeRegressor####


from sklearn.tree import DecisionTreeRegressor

DST_model = DecisionTreeRegressor()

param_grid = {
        'max_depth': [None, 10, 100],
        'min_samples_split': [2, 10, 50]
}

grid_search = GridSearchCV(estimator=DST_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)


print("DST Starting Grid Search...")
grid_search.fit(x, y)


print("Best parameters found: ", grid_search.best_params_)


best_DST_model = DecisionTreeRegressor(**grid_search.best_params_)


print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_DST_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_DST_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)


avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)


print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)


print("Training the best model...")
best_DST_model.fit(x, y)

print("Evaluating the best model on all set...")
y_pred = best_DST_model.predict(x)

test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)

print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_DST_model.pkl'
joblib.dump(best_DST_model, model_path)
print(f"DST Best model saved to {model_path}")

REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_DST_model = joblib.load('best_DST_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_DST_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_DST_importance.csv', index=False)



####RandomForestRegressor####

from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor()
param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10],
        'min_samples_split': [2, 10, 50]
}


grid_search = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)

print("RF Starting Grid Search...")
grid_search.fit(x, y)

print("Best parameters found: ", grid_search.best_params_)

best_RF_model = RandomForestRegressor(**grid_search.best_params_)

print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_RF_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_RF_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)


avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)

print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)

print("Training the best model...")
best_RF_model.fit(x, y)

print("Evaluating the best model on the all set...")
y_pred = best_RF_model.predict(x)

test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)

print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_RF_model.pkl'
joblib.dump(best_RF_model, model_path)
print(f"RF Best model saved to {model_path}")


REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_RF_model = joblib.load('best_RF_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_RF_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_RF_importance.csv', index=False)





####GradientBoostingRegressor####

from sklearn.ensemble import GradientBoostingRegressor


GB_model = GradientBoostingRegressor()

param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1]
}


grid_search = GridSearchCV(estimator=GB_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)


print("Starting Grid Search...")
grid_search.fit(x, y)


print("Best parameters found: ", grid_search.best_params_)


best_GB_model = GradientBoostingRegressor(**grid_search.best_params_)


print("Evaluating model with 10 times 5-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_GB_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_GB_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)


avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)


print("Average MSE: ", avg_mse)
print("Average R²: ", avg_r2)


print("Training the best model...")
best_GB_model.fit(x, y)

print("Evaluating the best model on the all set...")
y_pred = best_GB_model.predict(x)


test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)

print("Test MSE: ", test_mse)
print("Test R²: ", test_r2)

model_path = 'best_GB_model.pkl'
joblib.dump(best_GB_model, model_path)
print(f"Best model saved to {model_path}")

REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_GB_model = joblib.load('best_GB_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_GB_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_GB_importance.csv', index=False)





####KNeighborsRegressor####

from sklearn.neighbors import KNeighborsRegressor

KN_model = KNeighborsRegressor()

param_grid = {
        'n_neighbors': [3, 5, 9],
        'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(estimator=KN_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)

print("KN Starting Grid Search...")
grid_search.fit(x, y)

print("Best parameters found: ", grid_search.best_params_)

best_KN_model = KNeighborsRegressor(**grid_search.best_params_)

print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_KN_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_KN_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)


avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)

print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)

print("Training the best model...")
best_KN_model.fit(x, y)

print("Evaluating the best model on the test set...")
y_pred = best_KN_model.predict(x)

test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)

print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_KN_model.pkl'
joblib.dump(best_KN_model, model_path)
print(f"KN Best model saved to {model_path}")


REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_KN_model = joblib.load('best_KN_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_KN_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_KN_importance.csv', index=False)




#######SVR######


from sklearn.svm import SVR

SVR_model = SVR()

param_grid = {
        'C': [0.1, 0.5, 1.0],
        'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(estimator=SVR_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)

print("Starting Grid Search...")
grid_search.fit(x, y)

print("SVR Best parameters found: ", grid_search.best_params_)

best_SVR_model = SVR(**grid_search.best_params_)

print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_SVR_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_SVR_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)

avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)

print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)

print("Training the best model...")
best_SVR_model.fit(x, y)

print("Evaluating the best model on the test set...")
y_pred = best_SVR_model.predict(x)

test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)

print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_SVR_model.pkl'
joblib.dump(best_SVR_model, model_path)
print(f"SVR Best model saved to {model_path}")

REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_SVR_model = joblib.load('best_SVR_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_SVR_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_SVR_importance.csv', index=False)




####MLPRegressor####

from sklearn.neural_network import MLPRegressor

MLP_model = MLPRegressor()

param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (500,)],
        'activation': ['relu', 'tanh']
}

grid_search = GridSearchCV(estimator=MLP_model, param_grid=param_grid, cv=6, scoring='neg_mean_squared_error', verbose=2)

print("MLP Starting Grid Search...")
grid_search.fit(x, y)

print("Best parameters found: ", grid_search.best_params_)

best_MLP_model = MLPRegressor(**grid_search.best_params_)

print("Evaluating model with 100 times 6-fold cross-validation...")
mse_scores = []
r2_scores = []

for i in range(100):
    kf = KFold(n_splits=6, shuffle=True)
    fold_mse_scores = cross_val_score(best_MLP_model, x, y, cv=kf, scoring='neg_mean_squared_error', verbose=2)
    fold_r2_scores = cross_val_score(best_MLP_model, x, y, cv=kf, scoring='r2', verbose=2)
    mse_scores.extend(fold_mse_scores)
    r2_scores.extend(fold_r2_scores)

avg_mse = np.mean(-np.array(mse_scores))
avg_r2 = np.mean(r2_scores)

print("Average MSE: ", avg_mse)
print("Average R2: ", avg_r2)

print("Training the best model...")
best_MLP_model.fit(x, y)

print("Evaluating the best model on the test set...")
y_pred = best_MLP_model.predict(x)

test_mse = mean_squared_error(y, y_pred)
test_r2 = r2_score(y, y_pred)

print("Test MSE: ", test_mse)
print("Test R2: ", test_r2)

model_path = 'best_MLP_model.pkl'
joblib.dump(best_MLP_model, model_path)
print(f"MLP Best model saved to {model_path}")

REPEAT_TIMES = 100
COLS = list(X1.columns)

gpu_strategy = tf.distribute.get_strategy() 
with gpu_strategy.scope():

    best_MLP_model = joblib.load('best_MLP_model.pkl')
    

    all_results = []

    for _ in tqdm(range(REPEAT_TIMES)):
        results = []

        for k in range(len(COLS)):
            if k > 0: 
                save_col = x[:, k-1].copy()
                np.random.shuffle(x[:, k-1])
           
                oof_preds = best_MLP_model.predict(x)
                mae_score = np.mean(np.abs(oof_preds - y))
                results.append(mae_score)
     
                if k > 0: 
                    x[:, k-1] = save_col

        all_results.append(results)


    avg_results = np.mean(all_results, axis=0)

    avg_results_df = pd.DataFrame({'feature': COLS[1:], 'avg_mae': avg_results})
    avg_results_df = avg_results_df.sort_values('avg_mae', ascending=False)
    print(avg_results_df)

    avg_results_df.to_csv('feature_MLP_importance.csv', index=False)

