import pandas as pd
import xgboost as xgb
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

from sklearn.utils import shuffle

out_path = '../output/stacking/'

df_train = pd.read_csv('../data/train_stacking.csv')
df_test = pd.read_csv('../data/test_stacking.csv')

train_y = df_train['is_churn']
train_x = df_train.drop(['msno', 'is_churn'], axis=1)

test_x = df_test.drop(['msno'], axis=1)

bst = None
folds = 10
pred = None

cv_scores = []

train_x, train_y = shuffle(train_x, train_y)

kf = KFold(n=train_x.shape[0], n_folds=folds, shuffle=True, random_state=2017)
for i, (train_index, test_index) in enumerate(kf):

    X_train, X_val = np.array(train_x)[train_index], np.array(train_x)[test_index]
    y_train, y_val = np.array(train_y)[train_index], np.array(train_y)[test_index]

    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)

    del X_train, y_train
    gc.collect()

    evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]

    params = {
        'eta': 0.02,  # use 0.002
        'max_depth': 5,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }

    print('training...')
    bst = xgb.train(params, xgb_train, num_boost_round=150, evals=evallist, verbose_eval=50, maximize=False,
                        early_stopping_rounds=5)
    del xgb_train, xgb_val
    gc.collect()

    cv_scores.append(roc_auc_score(y_val, bst.predict(xgb.DMatrix(X_val), ntree_limit=bst.best_ntree_limit)))
    print(cv_scores)

    print('predicting...')
    if i == 0:
        pred = bst.predict(xgb.DMatrix(np.array(test_x)),
                               ntree_limit=bst.best_ntree_limit)
    else:
        pred += bst.predict(xgb.DMatrix(np.array(test_x)),
                                ntree_limit=bst.best_ntree_limit)

del train_x, train_y
gc.collect()

print('mean_score:', np.mean(cv_scores))

pred /= folds
df_test['is_churn'] = pred.clip(0.0000001, 0.999999)
df_test = df_test[['msno', 'is_churn']]

# df_test.to_csv(out_path + 'stack_submissions{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
df_test = []

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=bst)
plt.show()
# plt.savefig('./feature_importance.png', dpi=100)
