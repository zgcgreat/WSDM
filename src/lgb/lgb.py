import gc

from sklearn.ensemble import GradientBoostingClassifier

gc.enable()
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn_pandas import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost.sklearn import XGBClassifier

start = datetime.now()

in_path = '../../data/'
out_path = '../../output/lgb/'


df_train = pd.read_csv(in_path + 'train.csv')

drop_list = []
df_train = df_train.drop(drop_list, axis=1)

df_test = pd.read_csv(in_path + 'train_v2.csv')
df_test = df_test.drop(drop_list, axis=1)
print(df_test.columns)

label_train = df_train['is_churn']
train_data = df_train.drop(['msno', 'is_churn', 'msno_hash'], axis=1)

label_test = df_test['is_churn']
test_data = df_test.drop(['msno', 'is_churn', 'msno_hash'], axis=1)

del df_train, df_test
gc.collect()

print(len(train_data.columns), len(test_data.columns))

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
# clf.fit(train_data, label_train)
# new_feature = clf.apply(train_data)
# new_feature = pd.DataFrame(new_feature)
# train_data = pd.concat([train_data, new_feature], axis=1)
# new_feature = clf.apply(test_data)
# new_feature = pd.DataFrame(new_feature)
# test_data = pd.concat([test_data, new_feature], axis=1)
# print(train_data.head())


print('splitting data...')
train_data, label_train = shuffle(train_data, label_train)
lgb_train = lgb.Dataset(train_data, label_train)
lgb_val = lgb.Dataset(test_data, label_test)

# del X_train, y_train, X_val, y_val
# gc.collect()

params = {
            'objective': 'regression',
            'metric': ['binary_logloss'],
            'boosting': 'gbdt',
            'learning_rate': 0.1,
            'verbose': 1,
            'num_leaves': 2*7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.8,
            'feature_fraction_seed': 1,
            'use_two_round_loading': 'true',
            'is_unbalance': True,
            'max_bin': 255,
            'max_depth': 7,
            'is_training_metric': True,
            'nthread': 8
    }
print('training...')
bst = lgb.train(params, lgb_train, num_boost_round=150, verbose_eval=5, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=5)
del lgb_train, lgb_val
gc.collect()

# del train_data, label_train
# gc.collect()


# plt.rcParams['figure.figsize'] = (7.0, 7.0)
# lgb.plot_importance(booster=bst)
# plt.show()
# plt.savefig('./feature_importance.jpg', dpi=100)

# estimator = lgb.LGBMRegressor(num_leaves=31)
# #
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'max_depth': [5, 7, 10, 12, 15]
# }
#
# gbm = GridSearchCV(estimator, param_grid)
#
# gbm.fit(pd.DataFrame(train_data), label_train)
#
# print('Best parameters found by grid search are:', gbm.best_params_)

print(datetime.now() - start)

'''
['city', 'bd', 'gender', 'registered_via', 'registration_init_year', 'registration_init_month', 'registration_init_date', 
 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'transaction_date', 
 'membership_expire_date', 'is_cancel', 'transaction_date_year', 'transaction_date_month', 'transaction_date_date', 
 'membership_expire_date_year', 'membership_expire_date_month', 'membership_expire_date_date', 'trans_count', 
 'membership_transaction_gap', 'discount', 'is_discount', 'amt_per_day', 'autorenew_&_not_cancel', 
 'notAutorenew_&_cancel', 'total_order', 'cancel_times', 'total_actual_payment', 'plan_net_worth', 'auto_renew_count', 
 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs', 'avg_num_50', 'avg_total_secs', 'avg_num_25', 
 'avg_num_75', 'avg_num_unq', 'avg_num_985', 'logs_count', 'avg_num_100', 'user_date_year', 'user_date_month', 'user_date_date'] 
 
['city', 'bd', 'gender', 'registered_via', 'registration_init_year', 'registration_init_month', 'registration_init_date', 
 'payment_method_id', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'transaction_date', 
 'membership_expire_date', 'is_cancel', 'transaction_date_year', 'transaction_date_month', 'transaction_date_date', 
 'membership_expire_date_year', 'membership_expire_date_month', 'membership_expire_date_date', 'trans_count',
  'membership_transaction_gap', 'discount', 'is_discount', 'amt_per_day', 'autorenew_&_not_cancel', 
  'notAutorenew_&_cancel', 'total_order', 'cancel_times', 'total_actual_payment', 'plan_net_worth', 'auto_renew_count', 
  'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs', 'avg_num_25', 'logs_count', 'avg_num_unq', 
  'avg_num_985', 'avg_total_secs', 'avg_num_50', 'avg_num_100', 'avg_num_75', 'user_date_year', 'user_date_month', 'user_date_date']
'''