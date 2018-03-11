from multiprocessing import Pool, cpu_count
import gc;

from xgboost import XGBClassifier

gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

in_path = '../../data_ori/'

# train = pd.read_csv(in_path + 'train.csv')
# test = pd.read_csv(in_path + 'sample_submission_zero.csv')
#
# transactions = pd.read_csv(in_path + 'transactions.csv', usecols=['msno'])
# transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
# transactions.columns = ['msno', 'trans_count']
# train = pd.merge(train, transactions, how='left', on='msno')
# test = pd.merge(test, transactions, how='left', on='msno')
# transactions = [];
# print('transaction merge...')
#
# user_logs = pd.read_csv(in_path + 'user_logs.csv', usecols=['msno'])
# user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
# user_logs.columns = ['msno', 'logs_count']
# train = pd.merge(train, user_logs, how='left', on='msno')
# test = pd.merge(test, user_logs, how='left', on='msno')
# user_logs = [];
# print('user logs merge...')
#
# members = pd.read_csv(in_path + 'members.csv')
# train = pd.merge(train, members, how='left', on='msno')
# test = pd.merge(test, members, how='left', on='msno')
# members = [];
# print('members merge...')
#
# gender = {'male': 1, 'female': 2}
# train['gender'] = train['gender'].map(gender)
# test['gender'] = test['gender'].map(gender)
#
# train = train.fillna(0)
# test = test.fillna(0)
#
# transactions = pd.read_csv(in_path + 'transactions.csv')
# transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
# transactions = transactions.drop_duplicates(subset=['msno'], keep='first')
#
# train = pd.merge(train, transactions, how='left', on='msno')
# test = pd.merge(test, transactions, how='left', on='msno')
# transactions = []
#
#
# def transform_df(df):
#     df = pd.DataFrame(df)
#     df = df.sort_values(by=['date'], ascending=[False])
#     df = df.reset_index(drop=True)
#     df = df.drop_duplicates(subset=['msno'], keep='first')
#     return df
#
#
# def transform_df2(df):
#     df = df.sort_values(by=['date'], ascending=[False])
#     df = df.reset_index(drop=True)
#     df = df.drop_duplicates(subset=['msno'], keep='first')
#     return df
#
#
# df_iter = pd.read_csv(in_path + 'user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
# last_user_logs = []
# i = 0  # ~400 Million Records - starting at the end but remove locally if needed
# for df in df_iter:
#     if i > 35:
#         if len(df) > 0:
#             print(df.shape)
#             p = Pool(cpu_count())
#             df = p.map(transform_df, np.array_split(df, cpu_count()))
#             df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
#             df = transform_df2(df)
#             p.close();
#             p.join()
#             last_user_logs.append(df)
#             print('...', df.shape)
#             df = []
#     i += 1
#
# last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
# last_user_logs = transform_df2(last_user_logs)
#
# train = pd.merge(train, last_user_logs, how='left', on='msno')
# test = pd.merge(test, last_user_logs, how='left', on='msno')
# last_user_logs = []
#
# train = train.fillna(0)
# test = test.fillna(0)
#
# train.to_csv('../../data/train1.csv', index=False)
# test.to_csv('../../data/test1.csv', index=False)


train = pd.read_csv('../../data/best_model_train.csv')
test = pd.read_csv('../../data/best_model_test.csv')
train_y = pd.read_csv('../../data/train.csv', usecols=['is_churn'])
cols = [c for c in train.columns if c not in ['is_churn', 'msno']]

print(train.columns)

train_x = train[cols]
# train_y = train['is_churn']


# 结果变差了一点0.22515
# clf = XGBClassifier(
#     learning_rate=0.3,  # 默认0.3
#     n_estimators=30,  # 树的个数
#     max_depth=3,
#     min_child_weight=1,
#     gamma=0.5,
#     subsample=0.6,
#     colsample_bytree=0.6,
#     objective='binary:logistic',  # 逻辑回归损失函数
#     nthread=8,  # cpu线程数
#     scale_pos_weight=1,
#     reg_alpha=1e-05,
#     reg_lambda=1,
#     seed=2017)  # 随机种子
#
# clf.fit(train_x, train_y)
# new_feature = clf.apply(train_x)
# new_feature = pd.DataFrame(new_feature)
# train_x = pd.concat([train_x, new_feature], axis=1)
# new_feature = clf.apply(test.drop(['msno', 'is_churn'], axis=1))
# new_feature = pd.DataFrame(new_feature)
# test_x = pd.concat([test.drop(['msno', 'is_churn'], axis=1), new_feature], axis=1)


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)


fold = 5
for i in range(fold):
    params = {
        'eta': 0.02,  # use 0.002
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train_x, train_y, test_size=0.3, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 2000, watchlist, feval=xgb_score, maximize=False, verbose_eval=50,
                      early_stopping_rounds=50)  # use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test.drop(['msno'], axis=1)), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test.drop(['msno'], axis=1)), ntree_limit=model.best_ntree_limit)
pred /= fold
test['is_churn'] = pred.clip(0.0000001, 0.999999)
test[['msno', 'is_churn']].to_csv(
    '../../output/xgb/xgb_lr_submission{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model)
plt.show()
