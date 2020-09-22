# coding: utf-8

import sys
import time

import lightgbm as lgb
import sklearn.metrics as metrics

from dota_data import load_as_data_frame

# Load data and build LightGBM dataset.
time_start = time.perf_counter()
df, _ = load_as_data_frame(sys.argv[1], sets=('train', 'valid', 'test'))
train_data = lgb.Dataset(df['train'].drop('result', axis=1), df['train'].result)
valid_data = lgb.Dataset(df['valid'].drop('result', axis=1), df['valid'].result, reference=train_data)
print(f'Data loading time: {(time.perf_counter() - time_start) / 60.0 : .2f} min')

# Specify hyper-parameters.
params = {
    'colsample_bytree': 0.8,  # Sample 80% of the columns when training each tree.
    'learning_rate': 0.1,
    'metric': 'binary_error',  # Use accuracy as the validation metric.
    'min_child_samples': 600,  # Each leaf node must contain at least 600 rows (samples).
    'num_leaves': 60,  # Maximum number of leaf nodes of each tree.
    'objective': 'binary',  # Use loss function of binary classification (binary cross entropy loss).
    'reg_alpha': 0.0,
    'reg_lambda': 0.3,  # L2 regularization factor for the weights of the leaf.
    'subsample': 0.05,  # Sample 5% of the rows when training each tree.
    'num_iterations': 300,  # Maximum number of trees.
}

# Train the LightGBM model.
time_start = time.perf_counter()
m = lgb.train(params, train_data, early_stopping_rounds=100, valid_sets=[valid_data], verbose_eval=True)
print(f'Training time: {(time.perf_counter() - time_start) / 60.0 : .2f} min')

# Save the model to "model.txt"
m.save_model(sys.argv[2])

# Compute and report the accuracy on the test set.
pred = m.predict(df['test'].drop('result', axis=1))
print(f'test acc = {metrics.accuracy_score(df["test"].result, pred >= 0.5)}')
