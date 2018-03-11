import gc
gc.enable()
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
import hashlib
warnings.filterwarnings('ignore')


start = datetime.now()

in_path = '../../data_ori/'
out_path = '../../data/'


def mem_use(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(index=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(index=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    # print("{:03.2f} MB".format(usage_mb))
    return int(usage_mb)


# 多线程并行计算
# def applyParallel(df, func):
#     with Parallel(n_jobs=8) as parallel:
#         retLst = parallel(delayed(function=func)(pd.Series(value)) for key, value in df.iterrows())
#         return pd.concat(retLst, axis=0)


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        return df


def convert_age(age):
    return int(int(age) / 8)


# members
def get_members(df_members):
    gender = {'male': 1, 'female': 2, np.NaN: 0}

    df_members['registration_init_year'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[:4])).astype(
        np.int16)
    df_members['registration_init_month'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[4:6])).astype(
        np.int8)
    df_members['registration_init_date'] = df_members['registration_init_time'].apply(lambda x: int(str(x)[-2:])).astype(
        np.int8)

    # --- Now drop the unwanted date columns ---
    # df_members = df_members.drop('registration_init_time', axis=1)
    df_members['gender'] = df_members['gender'].map(gender).astype(np.int8)
    df_members['bd'] = df_members['bd'].clip(0, 100)  # 限制年龄在0-100之间
    df_members['bd'] = df_members['bd'].apply(convert_age)
    df_members['bd'] = df_members['bd'].astype(np.int8)

    return df_members


def get_gap(expire_date, transaction_date):
    gap = (expire_date[:4] * 365 + expire_date[4:6] * 30 + expire_date[6:]) \
          - (transaction_date[:4] * 365 + transaction_date[4:6] * 30 + transaction_date[6:])
    return gap


# transactions
def get_transactions(df_transactions):
    df_transactions['transaction_date_year'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[2:4])).astype(
        np.int16)
    df_transactions['transaction_date_month'] = df_transactions['transaction_date'].apply(
        lambda x: int(str(x)[4:6])).astype(np.int8)
    df_transactions['transaction_date_date'] = df_transactions['transaction_date'].apply(lambda x: int(str(x)[-2:])).astype(
        np.int8)
    df_transactions['membership_expire_date_year'] = df_transactions['membership_expire_date'].apply(
        lambda x: int(str(x)[2:4])).astype(np.int16)
    df_transactions['membership_expire_date_month'] = df_transactions['membership_expire_date'].apply(
        lambda x: int(str(x)[4:6])).astype(np.int8)
    df_transactions['membership_expire_date_date'] = df_transactions['membership_expire_date'].apply(
        lambda x: int(str(x)[-2:])).astype(np.int8)
    df_tmp = df_transactions['msno'].value_counts().reset_index()
    df_tmp.columns = ['msno', 'trans_count']
    df_transactions = pd.merge(left=df_transactions, right=df_tmp, how='left', on='msno')

    # 会员到期与交易日期的时间差
    df_transactions['membership_transaction_gap'] = (df_transactions['membership_expire_date_year'] * 365 +
                                                     df_transactions['membership_expire_date_month'] * 30 +
                                                     df_transactions['membership_expire_date_date']) - \
                                                    (df_transactions['transaction_date_year'] * 365 +
                                                     df_transactions['transaction_date_month'] * 30 +
                                                     df_transactions['transaction_date_date'])

    df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']
    df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
    df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']

    df_transactions['autorenew_&_not_cancel'] = ((df_transactions['is_auto_renew']==1) == (df_transactions['is_cancel']==0)).astype(np.int8)
    df_transactions['notAutorenew_&_cancel'] = ((df_transactions['is_auto_renew'] == 0) == (df_transactions['is_cancel'] == 1)).astype(np.int8)

    df_transactions_train = df_transactions.loc[df_transactions['transaction_date'] < 20170301.]
    df_transactions_test = df_transactions.loc[df_transactions['transaction_date'] < 20170401.]

    df_transactions_train = df_transactions_train.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
    df_transactions_test = df_transactions_test.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)

    df_transactions_train = df_transactions_train.drop_duplicates(subset=['msno'], keep='first')
    df_transactions_test = df_transactions_test.drop_duplicates(subset=['msno'], keep='first')

    df_transactions = []
    del df_transactions
    gc.collect()

    df_transactions_train = df_transactions_train.sort_values(by=['transaction_date'], ascending=False).reset_index(drop=True)
    df_transactions_train = df_transactions_train.drop_duplicates(subset=['msno'], keep='first')

    df_transactions_test = df_transactions_test.sort_values(by=['transaction_date'], ascending=False).reset_index(drop=True)
    df_transactions_test = df_transactions_test.drop_duplicates(subset=['msno'], keep='first')

    # --- Now drop the unwanted date columns ---
    # df_transactions_train = df_transactions_train.drop(['transaction_date', 'membership_expire_date'], axis=1)
    # df_transactions_test = df_transactions_test.drop(['transaction_date', 'membership_expire_date'], axis=1)

    print('memery use of df_transactions_train:{:03.2f} MB'.format(mem_use(df_transactions_train)))
    print('memery use of df_transactions_test:{:03.2f} MB'.format(mem_use(df_transactions_test)))
    return df_transactions_train, df_transactions_test


# 统计特征
def get_transaction_stats(df_transactions):
    grouped = df_transactions.copy().groupby('msno')
    df_stats = grouped.agg({'msno': {'total_order': 'count'},
                            'plan_list_price': {'plan_net_worth': 'sum'},
                            'actual_amount_paid': {'total_actual_payment': 'sum'},
                            'is_auto_renew': {'auto_renew_count': 'sum'},
                            'is_cancel': {'cancel_times': lambda x: sum(x == 1)}})
    df_stats.columns = df_stats.columns.droplevel(0)
    df_stats = df_stats.reset_index()
    return df_stats


def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=False)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df


def get_churn_rate(df_train, feat):
    df = df_train.groupby(feat).apply(lambda df: np.mean(df['is_churn'])).reset_index()
    df.columns = [feat, '{}_churn_rate'.format(feat)]
    return df


def merge_df(df, df_rate):
    for t, feat in enumerate(df_train.columns, start=1):
        if feat not in ['msno', 'msno_hash', 'is_churn']:
            df_rate = pd.merge(left=df_rate, right=df_rate, how='left', on=feat)

    return df


if __name__=='__main__':
    df_members = pd.read_csv(in_path + 'members_v3.csv',
                             dtype={'city': np.int8, 'bd': np.int16, 'registered_via': np.int8, 'gender': 'object'})
    df_members = parallel(df_members, get_members)
    print('memery use of df_members:{:03.2f} MB'.format(mem_use(df_members)))

    df_transactions = pd.read_csv(in_path + 'transactions.csv', dtype={'payment_method': np.int8, 'payment_plan_days': np.int8,
        'plan_list_price': np.int16, 'actual_amount_paid': np.int16, 'is_auto_renew': np.int8, 'is_cancel': np.int8})
    df_transactions_v2 = pd.read_csv(in_path + 'transactions_v2.csv',
                                  dtype={'payment_method': np.int8, 'payment_plan_days': np.int8,
                                         'plan_list_price': np.int16, 'actual_amount_paid': np.int16,
                                         'is_auto_renew': np.int8, 'is_cancel': np.int8})
    df_transactions = pd.concat([df_transactions, df_transactions_v2], axis=0)
    df_transactions = df_transactions.drop_duplicates()
    df_transactions_v2 = []
    df_transactions_train, df_transactions_test = get_transactions(df_transactions)


    # 应用多线程并行统计
    df_stats_train = parallel(df_transactions_train, get_transaction_stats)
    df_stats_test = parallel(df_transactions_test, get_transaction_stats)
    # df_stats_train = get_transactions(df_transactions_train)
    # df_stats_test = get_transactions(df_transactions_test)

    print('memery use of df_stats_train:{:03.2f} MB'.format(mem_use(df_stats_train)))

    df_train = pd.read_csv(in_path + 'train.csv', dtype={'is_churn': np.int8, 'mnso': str})
    df_train = pd.merge(left=df_train, right=df_members, how='left', on=['msno'])
    df_train = pd.merge(left=df_train, right=df_transactions_train, how='left', on='msno')
    df_train = pd.merge(left=df_train, right=df_stats_train, how='left', on='msno')
    # 释放内存
    df_stats_train = []
    df_transactions_train = []
    del df_transactions_train, df_stats_train
    gc.collect()

    # last_user_logs = get_user_logs()
    last_user_logs = pd.read_csv(out_path + 'user_log_train.csv')
    print('memery use of last_user_logs:{:03.2f} MB'.format(mem_use(last_user_logs)))
    df_train = pd.merge(left=df_train, right=last_user_logs, how='left', on='msno')
    df_train['msno_hash'] = df_train['msno'].apply(lambda x: hash(x))
    df_train.to_csv(out_path + 'train.csv', index=False)
    print('memery use of df_train:{:03.2f} MB'.format(mem_use(df_train)))

    df_train = []
    last_user_logs = []
    del df_train, last_user_logs
    gc.collect()

    last_user_logs = pd.read_csv(out_path + 'user_log_train_v2.csv')
    df_train_v2 = pd.read_csv(in_path + 'train_v2.csv', dtype={'is_churn': np.int8, 'mnso': str})
    df_train_v2 = pd.merge(left=df_train_v2, right=df_members, how='left', on=['msno'])
    df_train_v2 = pd.merge(left=df_train_v2, right=df_transactions_test, how='left', on='msno')
    df_train_v2 = pd.merge(left=df_train_v2, right=df_stats_test, how='left', on='msno')
    df_train_v2 = pd.merge(left=df_train_v2, right=last_user_logs, how='left', on='msno')
    df_train_v2['msno_hash'] = df_train_v2['msno'].apply(lambda x: hash(x))
    df_train_v2.to_csv(out_path + 'train_v2.csv', index=False)
    print('memery use of df_train:{:03.2f} MB'.format(mem_use(df_train_v2)))
    df_train_v2 = []
    last_user_logs = []
    del df_train_v2, last_user_logs
    gc.collect()

    df_test = pd.read_csv(in_path + 'sample_submission.csv', dtype={'mnso': str})
    df_test = pd.merge(left=df_test, right=df_members, how='left', on=['msno'])
    df_test = pd.merge(left=df_test, right=df_transactions_test, how='left', on='msno')
    df_test = pd.merge(left=df_test, right=df_stats_test, how='left', on='msno')
    #
    df_members = []
    df_stats_test = []
    df_transactions_test = []
    del df_transactions_test
    gc.collect()

    last_user_logs = pd.read_csv(out_path + 'user_log_test.csv')
    df_test = pd.merge(left=df_test, right=last_user_logs, how='left', on='msno')
    last_user_logs = []
    df_test['msno_hash'] = df_test['msno'].apply(lambda x: hash(x))
    df_test.to_csv(out_path + 'test.csv', index=False)
    print('memery use of df_test:{:03.2f} MB'.format(mem_use(df_test)))

    print(datetime.now() - start)
