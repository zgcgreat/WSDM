import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from sklearn.utils import shuffle
import hashlib
import pandas
import gc
from datetime import datetime

start = datetime.now()

path = '../../data_ori/'


def mem_use(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(index=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(index=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    # print("{:03.2f} MB".format(usage_mb))
    return int(usage_mb)


df_train = pd.read_csv(path + 'train.csv')


print(df_train.columns)
print('memery use of df_train:{:03.2f} MB'.format(mem_use(df_train)))

print(df_train.head(87331))
print(sum(df_train['is_churn']))
print(sum(df_train['is_churn']) / len(df_train))

print(datetime.now() - start)