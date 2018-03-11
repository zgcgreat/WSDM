import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn import preprocessing

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LinearRegression

# mnist = fetch_mldata('MNIST original', data_home='./tmp')

# only binary classification supported
# mask = (mnist['target'] == 3) + (mnist['target'] == 5)

def func(value):
    import math
    if value > 2:
        return int(math.log(float(value) ** 2))
    else:
        return abs(int(value))

cate_feats = ['city', 'bd', 'gender', 'registered_via', 'registration_init_year',
                  'registration_init_month', 'registration_init_date', 'expiration_date_year',
                  'expiration_date_month', 'expiration_date_date', 'payment_method_id', 'payment_plan_days',
                  'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel',
                  'transaction_date_year', 'transaction_date_month', 'transaction_date_date',
                  'membership_expire_date_year',
                  'membership_expire_date_month', 'membership_expire_date_date', 'membership_transaction_gap',
                  'cancel_times',
                  'auto_renew_count', 'total_order', 'plan_net_worth', 'user_date_year', 'user_date_month',
                  'user_date_date']

df_train = pd.read_csv('../../data/train.csv', nrows=100000)
df_test = pd.read_csv('../../data/test.csv', nrows=10000)
df_all = pd.concat([df_train, df_test], axis=0)
df_all = df_all.fillna(0).replace(np.inf, 0)
df_all = df_all.drop(['msno', 'is_churn', 'msno_hash'], axis=1)
y_train = df_train['is_churn']

for feat in df_all.columns:
    max_ = df_all[feat].max()
    df_all[feat] = (df_all[feat] - max_) * (-1)
    df_all[feat] = df_all[feat].apply(func)


df_all = df_all.astype('object')


# for i, f in enumerate(cate_feats):
#         if df_all[f].dtype == 'object':
#             lbl = preprocessing.LabelEncoder()
#             lbl.fit(list(df_all[f].values))
#             df_all[f] = lbl.transform(list(df_all[f].values))

# df_train = df_all.iloc[:df_train.shape[0], :]
# df_test = df_all.iloc[df_train.shape[0]:, :]
# drop_list = ['msno', 'is_churn', 'msno_hash']
# x_train = df_train.drop(drop_list, axis=1).values
# x_test = df_test.drop(drop_list, axis=1).values
# scale = StandardScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

enc = OneHotEncoder()
df_all = enc.fit_transform(df_all)
x_train = df_all[:df_train.shape[0]]
x_test = df_all[df_test.shape[0]:]

df_all = []


print('Dataset shape: {}'.format(x_train.shape))
print('Non-zeros rate: {}'.format(np.mean(x_train != 0)))
print('Classes balance: {} / {}'.format(np.mean(y_train == 0), np.mean(y_train == 1)))

X_tr, X_te, y_tr, y_te = train_test_split(x_train, y_train, random_state=42, test_size=0.3)


from tffm import TFFMClassifier, TFFMRegressor

for order in [3]:
    model = TFFMClassifier(
        order=order,
        rank=5,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.00001),
        n_epochs=5,
        batch_size=1024,
        init_std=0.001,
        reg=0.01,
        input_type='sparse',
        # session_config=tf.ConfigProto(log_device_placement=True, device_count={'GPU':0}),
        seed=42
    )
    model.fit(X_tr, y_tr, show_progress=True)
    predictions = model.predict(X_te)
    print('[order={}] logloss: {}'.format(order, log_loss(y_te, predictions)))
    print('[order={}] auc: {}'.format(order, roc_auc_score(y_te, predictions)))
    print(predictions)
    # this will close tf.Session and free resources
    model.destroy()
