import gc

from xgboost import XGBClassifier

gc.enable()
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn import *
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, StratifiedKFold
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

start = datetime.now()

in_path = '../../data/'
out_path = '../../output/xgb/'

drop_list = []

df_train = pd.read_csv(in_path + 'train_con.csv')
df_test = pd.read_csv(in_path + 'test_con.csv')

df_train = df_train.drop(drop_list, axis=1)
df_test = df_test.drop(drop_list, axis=1)
# df_train['is_curn'] = pd.read_csv(in_path + 'train.csv', usecols=['is_churn'])


# 可以自定义度量函数
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)


train_y = df_train['is_churn']
# train_y = pd.read_csv(in_path + 'train.csv', usecols=['is_churn'])
train_x = df_train.drop(['msno', 'is_churn'], axis=1)
test_x = df_test.drop(['msno', 'is_churn'], axis=1)

del df_train
gc.collect()

# clf = XGBClassifier(
#  learning_rate=0.3, #默认0.3
#  n_estimators=30, #树的个数
#  max_depth=3,
#  min_child_weight=1,
#  gamma=0.5,
#  subsample=0.6,
#  colsample_bytree=0.6,
#  objective='binary:logistic', #逻辑回归损失函数
#  nthread=8,  #cpu线程数
#  scale_pos_weight=1,
#  reg_alpha=1e-05,
#  reg_lambda=1,
#  seed=2017)  #随机种子
#
#
# clf.fit(train_x, train_y)
# new_feature = clf.apply(train_x)
# new_feature = pd.DataFrame(new_feature)
# train_x = pd.concat([train_x, new_feature], axis=1)
# new_feature = clf.apply(df_test.drop(['msno', 'is_churn'], axis=1))
# new_feature = pd.DataFrame(new_feature)
# test_x = pd.concat([df_test.drop(['msno', 'is_churn'], axis=1), new_feature], axis=1)

bst = None
folds = 5
pred = None
# skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2017)
kf = KFold(n=train_x.shape[0], n_folds=folds, shuffle=False, random_state=2017)
# for i, (train_index, test_index) in enumerate(kf, start=1):
for i in range(folds):
    # X_train, X_val = np.array(train_x)[train_index], np.array(train_x)[test_index]
    # y_train, y_val = np.array(train_y)[train_index], np.array(train_y)[test_index]
    train_x, train_y = shuffle(train_x, train_y)
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.3, random_state=i * 10)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)

    del X_train, y_train, X_val, y_val
    gc.collect()

    evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]

    params = {
        'eta': 0.02,  # use 0.002
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    # params = {'booster': 'gbtree',
    #               'learning_rate': 0.1,
    #               'n_estimators': 100,
    #               'bst:max_depth': 7,
    #               'bst:min_child_weight': 1,
    #               'bst:eta': 0.02,
    #               'silent': 1,
    #               'objective': 'binary:logistic',
    #               'gamma': 0.,
    #               'subsample': 0.8,
    #               'scale_pos_weight': 0.8,
    #               'colsample_bytree': 0.8,
    #               'eval_metric': ['logloss'],
    #               'nthread': 8,
    #               'seed': i}

    print('training...')
    bst = xgb.train(params, xgb_train, num_boost_round=150, evals=evallist, verbose_eval=1, maximize=False,
                        early_stopping_rounds=5)
    del xgb_train, xgb_val
    gc.collect()

    print('predicting...')
    if i == 0:
        pred = bst.predict(xgb.DMatrix(test_x),
                               ntree_limit=bst.best_ntree_limit)
    else:
        pred += bst.predict(xgb.DMatrix(test_x),
                                ntree_limit=bst.best_ntree_limit)

del train_x, train_y
gc.collect()

pred /= folds
df_test['is_churn'] = pred.clip(0.0000001, 0.999999)
df_test = df_test[['msno', 'is_churn']]

df_test.to_csv(out_path + 'xgb_submissions{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
df_test = []


plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=bst)
plt.show()
plt.savefig('./feature_importance.png', dpi=100)

print(datetime.now() - start)
