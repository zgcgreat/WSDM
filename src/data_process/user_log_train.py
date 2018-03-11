import pandas as pd
from csv import DictReader
from datetime import datetime
import warnings
import numpy as np
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')
import gc
gc.enable()

start = datetime.now()

in_path = '../../data_ori/'
out_path = '../../data/'
cache_path = '../../cache/'

size = 5000000


def parallel(df, func):
    if len(df) > 0:
        # print(df.shape)
        p = Pool(cpu_count())
        df = p.map(func, np.array_split(df, cpu_count()))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        return df


def get_unique_users():
    df_train = pd.read_csv(in_path + 'train.csv', usecols=['msno'])
    unique_users = set(df_train['msno'].unique())
    df_train = []
    del df_train
    gc.collect()
    return unique_users


# 去掉训练集和测试集中没有的用户的数据
def drop_user_log(unique_users):
    fo = open(cache_path + 'user_log_train.csv', 'w')
    fi = open(in_path + 'user_logs.csv', 'r')
    header = next(fi)
    fo.write(header)
    for t, row in enumerate(fi, start=1):
        if row.split(',')[0] in unique_users:
            fo.write(row)
        if t % 1000000 == 0:
            print(t)
    fi.close()
    fo.close()


def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=False)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df


def transform_date(df):
    df['user_date_year'] = df['date'].apply(lambda x: int(str(x)[2:4])).astype(np.int16)
    df['user_date_month'] = df['date'].apply(lambda x: int(str(x)[4:6])).astype(np.int8)
    df['user_date_date'] = df['date'].apply(lambda x: int(str(x)[-2:])).astype(np.int8)
    return df


def get_user_stats(df_user_log):
    grouped = df_user_log.groupby('msno')
    df_stats = grouped.agg({'msno': {'logs_count': 'count'},
                            'num_25': {'avg_num_25': 'mean'},
                            'num_50': {'avg_num_50': 'mean'},
                            'num_75': {'avg_num_75': 'mean'},
                            'num_985': {'avg_num_985': 'mean'},
                            'num_100': {'avg_num_100': 'mean'},
                            'num_unq': {'avg_num_unq': 'mean'},
                            'total_secs': {'avg_total_secs': 'mean'}})
    df_user_log = []; grouped = []
    del df_user_log, grouped
    gc.collect()
    df_stats.columns = df_stats.columns.droplevel(0)
    df_stats = df_stats.reset_index()
    return df_stats


def get_user_log():
    chunksize = size
    last_user_logs = []
    df_iter = pd.read_csv(cache_path + 'user_log_train.csv', iterator=True, chunksize=chunksize, dtype={'num_25': np.int8,
                                                                                              'num_50': np.int8,
        'num_75': np.int8, 'num_985': np.int8, 'num_100': np.int8, 'num_unq': np.int16})
    for i, df in enumerate(df_iter, start=1):
        df = df.drop_duplicates()
        df_stats = parallel(df, get_user_stats)
        df = parallel(df, transform_df)
        df = pd.merge(left=df, right=df_stats, how='left', on='msno')
        df_stats = []
        del df_stats
        gc.collect()
        last_user_logs.append(df)
        df = []
        del df
        gc.collect()
        print(i * chunksize)

    df = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
    last_user_logs = []
    del last_user_logs; gc.collect()
    df = transform_df(df)
    df = parallel(df, transform_date)
    df = df.drop(['date'], axis=1)
    df.to_csv(out_path + 'user_log_train.csv', index=False, chunksize=chunksize)


def func():
    chunksize = size
    df_all = pd.DataFrame()
    df_iter = pd.read_csv(cache_path + 'tmp.csv', iterator=True, chunksize=chunksize,
                          dtype={'num_25': np.int8,
                                 'num_50': np.int8,
                                 'num_75': np.int8, 'num_985': np.int8, 'num_100': np.int8, 'num_unq': np.int16})
    for i, df in enumerate(df_iter, start=1):
        df = transform_df(df)
        df = parallel(df, transform_date)
        df_all = df_all.append(df)
        df = []
        print(i * chunksize)
    df_all = transform_df(df_all)
    df_all = df_all.drop(['date'], axis=1)
    df_all.to_csv(out_path + 'user_log_train.csv', index=False)


if __name__ == '__main__':

    # unique_users = get_unique_users()
    # drop_user_log(unique_users)
    get_user_log()
    print(datetime.now())
