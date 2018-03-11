import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime


COLUMNS = ['msno', 'is_churn', 'city', 'bd', 'gender', 'registered_via',
       'registration_init_year', 'registration_init_month',
       'registration_init_date', 'expiration_date_year',
       'expiration_date_month', 'expiration_date_date',
       'expiration_registration_gap', 'payment_method_id', 'payment_plan_days',
       'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel',
       'transaction_date_year', 'transaction_date_month',
       'transaction_date_date', 'membership_expire_date_year',
       'membership_expire_date_month', 'membership_expire_date_date',
       'trans_count', 'membership_transaction_gap', 'plan_net_worth',
       'cancel_times', 'total_order', 'auto_renew_count',
       'total_actual_payment', 'num_25', 'num_50', 'num_75', 'num_985',
       'num_100', 'num_unq', 'total_secs', 'avg_total_secs', 'avg_num_985',
       'avg_num_100', 'logs_count', 'avg_num_25', 'avg_num_50', 'avg_num_75',
       'avg_num_unq', 'user_date_year', 'user_date_month', 'user_date_date',
       'msno_hash']

LABEL_COLUMN = "is_churn"

CATEGORICAL_COLUMNS = [
    'city', 'bd', 'gender', 'registered_via',
       'registration_init_year', 'registration_init_month',
       'registration_init_date', 'expiration_date_year',
       'expiration_date_month', 'expiration_date_date',
       'expiration_registration_gap', 'payment_method_id', 'payment_plan_days',
       'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel',
       'transaction_date_year', 'transaction_date_month',
       'transaction_date_date', 'membership_expire_date_year',
       'membership_expire_date_month', 'membership_expire_date_date',
       'trans_count', 'membership_transaction_gap', 'plan_net_worth',
       'cancel_times', 'total_order', 'auto_renew_count',
       'user_date_year', 'user_date_month', 'user_date_date',
       'msno_hash'
]

CONTINUOUS_COLUMNS = [
    'total_actual_payment', 'num_25', 'num_50', 'num_75', 'num_985',
       'num_100', 'num_unq', 'total_secs', 'avg_total_secs', 'avg_num_985',
       'avg_num_100', 'logs_count', 'avg_num_25', 'avg_num_50', 'avg_num_75',
       'avg_num_unq'
]


def load(filename):
    with open(filename, 'r') as f:
        skiprows = 1 if 'test' in filename else 0
        df = pd.read_csv(
            f, names=COLUMNS, skipinitialspace=True, skiprows=skiprows, engine='python'
        )
        df = df.dropna(how='any', axis=0)
    return df


def preprocess(df):
    df[LABEL_COLUMN] = df['is_churn']
    y = df[LABEL_COLUMN].values
    # df.pop(LABEL_COLUMN)
    df = df.drop(['msno', 'is_churn'], axis=1)
    for i, f in enumerate(CATEGORICAL_COLUMNS):
        if df[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))

    # df = pd.get_dummies(df, columns=[x for x in CATEGORICAL_COLUMNS])  # one-hot分类特征

    # TODO: select features for wide & deep parts
    
    # TODO: transformations (cross-products)
    # from sklearn.preprocessing import PolynomialFeatures
    # X = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)

    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    X = df.values
    return X, y


def wide_and_deep(X_train, y_train, X_val, y_val, X_test):

    wide = Sequential()
    wide.add(Dense(1, input_dim=X_train.shape[1]))
    
    deep = Sequential()
    # TODO: add embedding
    deep.add(Dense(input_dim=X_train.shape[1], output_dim=64, activation='relu'))
    deep.add(Dense(64, activation='relu'))
    deep.add(Dense(64, activation='relu'))
    deep.add(Dense(1, activation='sigmoid'))
    
    model = Sequential()
    model.add(Merge([wide, deep], mode='concat', concat_axis=1))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce = ReduceLROnPlateau(min_lr=0.0002, factor=0.05)
    model.fit([X_train, X_train], y_train,
              nb_epoch=1000,
              batch_size=32,
              validation_data=[[X_val, X_val], y_val],
              verbose=2,
              callbacks=[early_stopping, reduce])
    
    loss, accuracy = model.evaluate([X_val, X_val], y_val, verbose=2)
    print('val accuracy:', accuracy)
    print('val log_loss:', loss)
    pred = model.predict([X_test, X_test], verbose=0)
    return pred, loss
    
if __name__ == '__main__':

    df_train = pd.read_csv('../../data/train.csv')
    df_test = pd.read_csv('../../data/test.csv')
    df = pd.concat([df_train, df_test])
    train_len = len(df_train)

    X, y = preprocess(df)
    X_train = X[:train_len]
    y_train = y[:train_len]
    X_test = X[train_len:]
    # y_test = y[train_len:]

    # scale = StandardScaler()
    # scale.fit(X_train)
    # X_train = scale.transform(X_train)
    # X_test = scale.transform(X_test)

    folds = 5
    seed = 1
    pred = None
    cv_scores = []
    kf = KFold(X_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf):
        tr_x = X_train[train_index]
        tr_y = y_train[train_index]
        te_x = X_train[test_index]
        te_y = y_train[test_index]
        pred, loss = wide_and_deep(tr_x, tr_y, te_x, te_y, X_test)
        pred += pred
        cv_scores.append(loss)
        print('cv_scores:', cv_scores)

    pred = pred / folds
    df = pd.DataFrame()
    df['msno'] = df_test['msno'].values
    df['is_churn'] = pd.DataFrame(pred).clip(0.00001, 0.99999)
    df.to_csv('../../output/wnd/wnd_submission{}.csv'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), index=False)
    print('mean_cv_score:', np.mean(cv_scores))
    print(len(df))

