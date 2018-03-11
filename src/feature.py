import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score, log_loss, roc_auc_score
from datetime import datetime


class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


start = datetime.now()


def gen_features(train, val, test):
    train = pd.DataFrame(train)
    val = pd.DataFrame(val)
    test = pd.DataFrame(test)
    # cat_cols = ['city', 'bd', 'gender', 'registered_via', 'registration_init_year',
    #              'registration_init_month', 'registration_init_date', 'payment_method_id', 'payment_plan_days',
    #              'plan_list_price', 'actual_amount_paid', 'is_auto_renew', 'is_cancel',
    #              'transaction_date_year', 'transaction_date_month', 'transaction_date_date',
    #              'membership_expire_date_year',
    #              'membership_expire_date_month', 'membership_expire_date_date', 'membership_transaction_gap',
    #              'cancel_times',
    #              'auto_renew_count', 'plan_net_worth', 'user_date_year', 'user_date_month',
    #              'user_date_date']
    # con_cols = [x for x in train.columns if x not in cat_cols and x not in ['msno', 'is_churn']]
    # train[cat_cols] = train[cat_cols].astype('object')
    # test[cat_cols] = test[cat_cols].astype('object')
    # val[cat_cols] = val[cat_cols].astype('object')
    #
    # for col in cat_cols:
    #     train[col].fillna(value=train[col].mode()[0], inplace=True)
    #     test[col].fillna(value=test[col].mode()[0], inplace=True)
    #     val[col].fillna(value=val[col].mode()[0], inplace=True)
    # for col in con_cols:
    #     train[col].fillna(value=train[col].mean(), inplace=True)
    #     test[col].fillna(value=test[col].mean(), inplace=True)
    #     val[col].fillna(value=val[col].mean(), inplace=True)
    #
    # for c in train.columns:
    #     if train[c].dtype == 'object':
    #         lbl = LabelEncoder()
    #         lbl.fit(list(train[c].values) + list(test[c].values))
    #         train[c] = lbl.transform(list(train[c].values))
    #         test[c] = lbl.transform(list(test[c].values))

    n_comp = 15

    drop_list = []
    test_drop_list = []

    print(train.drop(drop_list, axis=1).shape, test.drop(test_drop_list, axis=1).shape)
    print('tSVD', datetime.now() - start)
    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp)
    tsvd_results_train = tsvd.fit_transform(train.drop(drop_list, axis=1))
    tsvd_results_val= tsvd.transform(val.drop(test_drop_list, axis=1))
    tsvd_results_test = tsvd.transform(test.drop(test_drop_list, axis=1))

    print('PCA', datetime.now() - start)
    # PCA
    pca = PCA(n_components=n_comp)
    pca2_results_train = pca.fit_transform(train.drop(drop_list, axis=1))
    pca2_results_val = pca.transform(val.drop(test_drop_list, axis=1))
    pca2_results_test = pca.transform(test.drop(test_drop_list, axis=1))

    print('ICA', datetime.now() - start)
    # ICA
    ica = FastICA(n_components=n_comp, max_iter=10000)
    ica2_results_train = ica.fit_transform(train.drop(drop_list, axis=1))
    ica2_results_val = ica.transform(val.drop(test_drop_list, axis=1))
    ica2_results_test = ica.transform(test.drop(test_drop_list, axis=1))

    print('GRP', datetime.now() - start)
    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1)
    grp_results_train = grp.fit_transform(train.drop(drop_list, axis=1))
    grp_results_val = grp.transform(val.drop(test_drop_list, axis=1))
    grp_results_test = grp.transform(test.drop(test_drop_list, axis=1))

    print('SRP', datetime.now() - start)
    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True)
    srp_results_train = srp.fit_transform(train.drop(drop_list, axis=1))
    srp_results_val = srp.transform(val.drop(test_drop_list, axis=1))
    srp_results_test = srp.transform(test.drop(test_drop_list, axis=1))

    # MCA
    # res_mca = MCA(train, ncp=n_comp, graph = FALSE)

    # save columns list before adding the decomposition components

    usable_columns = list(set(train.columns) - set(drop_list))

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        val['pca_' + str(i)] = pca2_results_val[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        val['ica_' + str(i)] = ica2_results_val[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]

        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        val['tsvd_' + str(i)] = tsvd_results_val[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        val['grp_' + str(i)] = grp_results_val[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]

        train['srp_' + str(i)] = srp_results_train[:, i - 1]
        val['srp_' + str(i)] = srp_results_val[:, i - 1]
        test['srp_' + str(i)] = srp_results_test[:, i - 1]
    return train, val, test



def gen_feature(train, test):
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    n_comp = 15
    drop_list = []
    test_drop_list = []

    print(train.drop(drop_list, axis=1).shape, test.drop(test_drop_list, axis=1).shape)
    print('tSVD', datetime.now() - start)
    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp)
    tsvd_results_train = tsvd.fit_transform(train.drop(drop_list, axis=1))
    tsvd_results_test = tsvd.transform(test.drop(test_drop_list, axis=1))

    print('PCA', datetime.now() - start)
    # PCA
    pca = PCA(n_components=n_comp)
    pca2_results_train = pca.fit_transform(train.drop(drop_list, axis=1))
    pca2_results_test = pca.transform(test.drop(test_drop_list, axis=1))

    print('ICA', datetime.now() - start)
    # ICA
    ica = FastICA(n_components=n_comp, max_iter=10000)
    ica2_results_train = ica.fit_transform(train.drop(drop_list, axis=1))
    ica2_results_test = ica.transform(test.drop(test_drop_list, axis=1))

    print('GRP', datetime.now() - start)
    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1)
    grp_results_train = grp.fit_transform(train.drop(drop_list, axis=1))
    grp_results_test = grp.transform(test.drop(test_drop_list, axis=1))

    print('SRP', datetime.now() - start)
    # SRP
    srp = SparseRandomProjection(n_components=n_comp, dense_output=True)
    srp_results_train = srp.fit_transform(train.drop(drop_list, axis=1))
    srp_results_test = srp.transform(test.drop(test_drop_list, axis=1))

    # MCA
    # res_mca = MCA(train, ncp=n_comp, graph = FALSE)

    # save columns list before adding the decomposition components

    usable_columns = list(set(train.columns) - set(drop_list))

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]

        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]

        train['srp_' + str(i)] = srp_results_train[:, i - 1]
        test['srp_' + str(i)] = srp_results_test[:, i - 1]
    return train, test