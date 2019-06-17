import Geohash as geo
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import itertools
from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.neighbors import DistanceMetric



def get_frac_sample(df, frac=0.2, random_seed=1):
    np.random.seed(random_seed)
    return df.sample(frac=0.2)


def preprocess(df_sample):
    """
        Basic Preprocessing for date, time and geo
    """
    df_sample['day'] = df_sample['day'].astype(np.int)
    df_sample['is_weekend'] = df_sample.day.map(lambda x: (max(x, 0) - 5) % 7 == 0 or (max(x, 0) - 6) % 7 == 0)
    df_sample['dow'] = df_sample.day.map(lambda x: (max(x, 0) - 6) % 7)
    df_sample['lat'] = df_sample.geohash6.map(lambda x: geo.decode(x)[0])
    df_sample['long'] = df_sample.geohash6.map(lambda x: geo.decode(x)[1])
    
    if 'timestamp' in df_sample:
        df_sample['hour'] = df_sample.timestamp.str.split(':').map(lambda x: x[0]).astype(np.int)
        df_sample['minute'] = df_sample.timestamp.str.split(':').map(lambda x: x[1]).astype(np.int)
    df_sample['hour_p8'] = (df_sample.hour + 8) % 24
    df_sample['is_working_hour'] = (df_sample.hour_p8 >= 9) & (df_sample.hour_p8 <= 18) & (df_sample.dow <=5)
    
    return df_sample


def draw_gmm_aic_bic_plot(x, max_component=10):
    aics = list()
    bics = list()
    components = [c for c in range(1, max_component)]
    for c in tqdm(components):
        mdl = GaussianMixture(c)
        mdl.fit(x)
        aics.append(mdl.aic(x))
        bics.append(mdl.bic(x))
    
    f, ax = plt.subplots(figsize=(10, 8))
    plt.title('AIC, BIC curve on GMM')
    plt.plot(components, aics, label='AIC', color='red')
    plt.plot(components, bics, label='BIC', color='blue')
    plt.legend()
    plt.show()


def add_cluster(cluster_mdl, df_agg, df_sample, cluster_column_name, cluster_proba_name, merge_by):
    cols = list()
    
    clusters = cluster_mdl.fit_predict(df_agg)
    
    if cluster_proba_name is not None:
        clusters_prob = np.apply_along_axis(
            lambda x: np.max(x), 1, cluster_mdl.predict_proba(df_agg)
        )
        df_agg[cluster_proba_name] = clusters_prob
        cols.append(cluster_proba_name)
    
    df_agg[cluster_column_name] = clusters
    cols.append(cluster_column_name)
        
    return df_sample.merge(
        df_agg[merge_by + cols]
        , how='inner'
        , left_on=merge_by
        , right_on=merge_by
    )


def rmse_per_region_geo(geohash6, y_true, y_hat, raw=False):
    """
        Calculate RMSE
    """
    err = np.square(y_true - y_hat)
    df_err = pd.DataFrame({
        'geohash6': geohash6,
        'mse': err,
        'y_true': y_true
    })
    df_err = df_err.groupby(['geohash6']).agg('mean')
    df_err['rmse'] = df_err.transform(np.sqrt)['mse']
    df_err['MAPE'] = (((df_err['mse'] + 1) / (df_err['y_true'] + 1) ) - 1) * 100
    return df_err.reset_index()

def rmse_per_region(lats, longs, y_true, y_hat, raw=False):
    err = np.square(y_true - y_hat)
    df_err = pd.DataFrame({
        'lat': lats,
        'long': longs,
        'mse': err,
        'y_true': y_true
    })
    df_err = df_err.groupby(['lat', 'long']).agg('mean')
    df_err['rmse'] = df_err.transform(np.sqrt)['mse']
    df_err['MAPE'] = (((df_err['mse'] + 1) / (df_err['y_true'] + 1) ) - 1) * 100
    return df_err.reset_index()
    
    


# +
def rmse(y_true, yhat):
    return np.sqrt(mean_squared_error(y_true, yhat))

def mape(y_true, yhat):
    return np.mean(np.abs(y_true - yhat) / y_true) * 100

def mape1(y_true, yhat):
    return np.mean((np.abs(y_true - yhat) + 1) /(y_true + 1)) * 100

def print_all_metrics(y_true, yhat):
    print(
        'mape: {}\nmape1: {}\nrmse: {}\nmae: {}\nr2: {}\neav: {}'.format(
            mape(y_true, yhat),
            mape1(y_true, yhat),
            rmse(y_true, yhat),
            mean_absolute_error(y_true, yhat),
            r2_score(y_true, yhat),
            explained_variance_score(y_true, yhat)
        )
    )


# -

def get_train_data(df, pct=0.3, n_days=None, start_day=1):
    unique_days = df.day.unique()
    
    if n_days is None:
        train_max_n_days = np.quantile(unique_days, pct)
    else:
        train_max_n_days = start_day + n_days
    
    return df[
        (df.day >= start_day)&
        (df.day <= train_max_n_days)
    ]


def fill_missing_timeframe(df, fill_with=0):
    all_possible_hours = [x for x in range(24)]
    all_possible_minutes = [x for x in range(0, 60, 15)]
    
    all_possible_timestamps = list(
        map(
            ':'.join, 
            list(
                itertools.product(
                    map(str, all_possible_hours), 
                    map(str, all_possible_minutes)
                )
            )
        )
    )
    df['day_timestamp'] = df.day.map(str) + ':' + df.timestamp
    unique_geohash6 = df.geohash6.unique()
    unique_days = df.day.unique()
    
    all_possible_daytimestamp = list(map(
        ':'.join,
        itertools.product(
            map(str, unique_days),
            all_possible_timestamps
        )
    ))
    
    for geohash6 in tqdm(unique_geohash6):
        place_daytimestamps = df[
            df.geohash6 == geohash6
        ]['day_timestamp'].values

        missing_day_timestamps = np.setdiff1d(
            all_possible_daytimestamp,
            place_daytimestamps,
            assume_unique=True
        )
        if len(missing_day_timestamps) > 0:
            df = df.append([
                {
                    'geohash6': geohash6,
                    'timestamp': ':'.join(day_timestamp.split(':')[1:]),
                    'day': day_timestamp.split(':')[0],
                    'demand': 0,
                    'day_timestamp': day_timestamp
                }
                for day_timestamp in missing_day_timestamps
            ], ignore_index=True)
        
    return df


# +
def generate_rolling_feature(df_train, window, group=['geohash6'], sort=['day', 'hour', 'minute']):
    return df_train.groupby(group).apply(
        lambda x: x.sort_values(by=sort)
    )['demand'].rolling(window).mean().reset_index().set_index('level_1')['demand']
    

def generate_demand_lag_feature(df_train, lag, group=['geohash6']):
    return df_train.groupby(group).shift(lag)['demand']
    
def more_preprocess(df_train, use_clustering=True):
    df_agg = df_train.groupby(['geohash6']).agg(['mean', 'median'])['demand']
    df_train['mean_demand_per_geo'] = df_train['geohash6'].map(df_agg['mean'])
    df_train['median_demand_per_geo'] = df_train['geohash6'].map(df_agg['median'])
    
    df_train['mean_demand_per_gdow'] = df_train.groupby(['geohash6', 'dow'])['demand'].transform('mean')
    df_train['median_demand_per_gdow'] = df_train.groupby(['geohash6', 'dow'])['demand'].transform('median')

    df_train['mean_demand_per_gdowh'] = df_train.groupby(['geohash6', 'dow', 'hour'])['demand'].transform('mean')
    df_train['median_demand_per_gdowh'] = df_train.groupby(['geohash6', 'dow', 'hour'])['demand'].transform('median')
    
    df_train['demand_tmin1'] = generate_demand_lag_feature(df_train, 1)
    df_train['demand_tmin4'] = generate_demand_lag_feature(df_train, 4)
    df_train['demand_tmin672'] = generate_demand_lag_feature(df_train, 672)
    
    # mean encoding
    df_train['mean_enc_demand_tmin1'] = df_train.groupby(['geohash6', 'dow', 'hour'])['demand_tmin1'].transform('mean')
    df_train['mean_enc_demand_tmin4'] = df_train.groupby(['geohash6', 'dow', 'hour'])['demand_tmin4'].transform('mean')
    df_train['mean_enc_demand_tmin672'] = df_train.groupby(['geohash6', 'dow', 'hour'])['demand_tmin672'].transform('mean')
    
    # quantiles
    df_train['demand_gdow_q95'] = df_train.groupby(['geohash6', 'dow'])['demand'].transform(lambda x: x.quantile(.95))
    df_train['demand_gdow_q10'] = df_train.groupby(['geohash6', 'dow'])['demand'].transform(lambda x: x.quantile(.1))
    
    # differencing
    df_train['demand_diff_wt_gdow_q95'] = df_train['demand'] - df_train['demand_gdow_q95']
    df_train['demand_diff_wt_gdow_q10'] = df_train['demand'] - df_train['demand_gdow_q10']
    
    # mean difference
    df_train['mean_demand_diff_wt_gdow_q95'] = df_train.groupby(['geohash6', 'dow'])['demand_diff_wt_gdow_q95'].transform('mean')
    df_train['mean_demand_diff_wt_gdow_q10'] = df_train.groupby(['geohash6', 'dow'])['demand_diff_wt_gdow_q10'].transform('mean')
    
    # simmple aggregation
    df_train['median_demand_per_gdowh'] = df_train['median_demand_per_gdowh'].fillna(0)
    df_train['mean_demand_per_gdowh'] = df_train['mean_demand_per_gdowh'].fillna(0)
    
    # cluster by location
    cluster_features = [
        'lat', 'long'
    ]
    df_cluster = df_train.groupby(cluster_features).agg('median').reset_index()[cluster_features]
    kmeans = KMeans(2)
    df_train = add_cluster(
            kmeans,
            df_cluster,
            df_train,
            'cluster_geo',
            None,
            cluster_features
        )
    mid_points = np.array(list(
        map(lambda x: kmeans.cluster_centers_[x], df_train['cluster_geo'])
    ))
    
    # find midpoints, assume it is center of city
    df_train['cluster_geo_mid_lat'] = mid_points[:, 0]
    df_train['cluster_geo_mid_long'] = mid_points[:, 1]
    
    # calculate distance to center
    dist = DistanceMetric.get_metric('haversine')
    df_train['dist_to_center_km'] = df_train.apply(
        lambda x: (dist.pairwise(np.radians([
            [x.lat, x.long],
            [x.cluster_geo_mid_lat, x.cluster_geo_mid_long]
        ])) * 6371)[1][0],
        axis=1
    )
    
    
    if use_clustering:
        cluster_features = [
            'lat', 'long', 'dow', 'hour'
        ]
        df_cluster = df_train.groupby(cluster_features).agg('mean')[[
            'median_demand_per_gdowh', 'mean_demand_per_gdowh'
        ]].reset_index()

        df_train = add_cluster(
            GaussianMixture(8),
            df_cluster,
            df_train,
            'cluster_gdowh',
            'cluster_prob_gdowh',
            cluster_features
        )
    
    # rolling feature
    # 1 hour
    df_train['demand_rolling_4'] = generate_rolling_feature(df_train, 4)
    
    return df_train



def get_future_n_ticks(day, hour, minute, n_ticks, tick_length_minutes=15):
    current_day, current_hour, current_minute = day, hour, minute
    res = list()
    for tick in range(n_ticks):
        current_minute += tick_length_minutes
        if current_minute >= 60:
            current_minute = current_minute % 60
            current_hour += 1
            
            if current_hour >= 24:
                current_hour = current_hour % 24
                current_day += 1
        res.append([current_day, current_hour, current_minute])
    return np.array(res)


def show_prediction_vs_actual_for_geo(pred_vs_act, geo_to_look_at, target_col='demand', pred_col='y_hat'):
    _d = pred_vs_act[
        pred_vs_act.geohash6 == geo_to_look_at
    ]
    _d = _d.sort_values(by=['day', 'hour', 'minute'])
    x = np.arange(_d.shape[0])
    sns.lineplot(
        x,
        _d[target_col],
        label='actual'
    )

    sns.lineplot(
        x,
        _d[pred_col],
        label='prediction'
    )
    plt.show()


def show_all_evaluation_metrics(future_res, test_set, geo_to_look_at = 'qp03wc'):
    future_res['timestamp'] = future_res['hour'].astype(str) + ':' + future_res['minute'].astype(str)
    pred_vs_act = test_set.merge(
        future_res,
        left_on=['geohash6', 'day', 'timestamp'],
        right_on=['geohash6', 'day', 'timestamp'],
        how='inner'
    )
    pred_evaluation = rmse_per_region_geo(
        pred_vs_act.geohash6,
        pred_vs_act.demand,
        pred_vs_act.y_hat
    )
    
    plt.title('RMSE distribution')
    sns.distplot(
        pred_evaluation.rmse
    )
    plt.show()

    plt.title('MAPE distribution')
    sns.distplot(
        pred_evaluation.MAPE
    )
    plt.show()
    
    print(
        'mean rmse: {}\nmean MAPE: {}'.format(
            pred_evaluation.rmse.mean(),
            pred_evaluation.MAPE.mean()
        )
    )
    
    _d = pred_vs_act[
        pred_vs_act.geohash6 == geo_to_look_at
    ]
    _d = _d.sort_values(by=['day', 'hour', 'minute'])
    x = np.arange(_d.shape[0])
    sns.lineplot(
        x,
        _d.demand,
        label='actual'
    )

    sns.lineplot(
        x,
        _d.y_hat,
        label='prediction'
    )
    plt.show()


class BenchmarkModel1(object):
    """
    Benchmark model that only predict with mean/median per geo, dow, hour, minute 
    """
    def __init__(self, strategy='mean'):
        self.df_train = None
        self.feature_columns = [
            'geohash6',
            'dow',
            'hour',
            'minute',
        ]
        self.strategy = strategy
        self.mdl = None
    
    def fit(self, X, y):
        _x = X.copy()
        _x['y_true'] = y
        self.mdl = _x.groupby(self.feature_columns).agg(self.strategy)['y_true'].reset_index()
    
    def fit_raw(self, df_train, target_column='demand'):
        self.df_train = preprocess(df_train)
        self.fit(
            self.df_train,
            self.df_train[target_column]
        )
    
    def predict(self, X):
        return X.merge(
            self.mdl,
            left_on=self.feature_columns,
            right_on=self.feature_columns,
            suffixes=('', '_r'),
            how='left'
        )['y_true']
        
    
    def predict_futures(self, n_ticks=5):
        temp_features = ['day', 'hour', 'minute']
        temp = self.df_train.groupby('geohash6')[temp_features + ['demand']].apply(
            lambda x: x.sort_values(by=temp_features, ascending=False).head(1)
        )
        future_rows=[]
        temp_arr = temp.reset_index().set_index('geohash6')[temp_features].apply(
            lambda x: get_future_n_ticks(*x,n_ticks), axis=1
        )
        temp_x = temp_arr.reset_index().apply(
            lambda r: [future_rows.append([r.geohash6] + list(t)) for t in r[0]], axis=1
        )
        # remove unused variables
        del temp_x
        future_df = pd.DataFrame(
            future_rows,
            columns=['geohash6', 'day', 'hour', 'minute']
        )
        
        x = preprocess(future_df)
        future_df['y_hat'] = self.predict(x)
        return future_df


class RegressorModel(object):
    """
    coupling the pipeline to be 1 model so it would simplify the process of training and 
    predicting esp. predicting the futures
    """
    def __init__(self, 
                 categorical_features=[
                     'is_weekend','dow','hour','cluster_gdowh', 'is_working_hour', 'cluster_geo'
                 ], 
                 numerical_features=[
                     'lat','long','minute','mean_demand_per_geo',
                    'median_demand_per_geo','mean_demand_per_gdow','median_demand_per_gdow','mean_enc_demand_tmin1',
                    'mean_enc_demand_tmin4','mean_enc_demand_tmin672','demand_gdow_q95','demand_gdow_q10',
                    'mean_demand_diff_wt_gdow_q95','mean_demand_diff_wt_gdow_q10','cluster_prob_gdowh','demand_rolling_4',
                     'dist_to_center_km'
                ], 
                 attributed_by=['lat', 'long', 'dow', 'hour'],
                 target_column='demand'):
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.attributions = ['cluster_gdowh', 'cluster_geo'] + [
            'mean_demand_per_geo',
            'median_demand_per_geo','mean_demand_per_gdow','median_demand_per_gdow','mean_enc_demand_tmin1',
            'mean_enc_demand_tmin4','mean_enc_demand_tmin672','demand_gdow_q95','demand_gdow_q10',
            'mean_demand_diff_wt_gdow_q95','mean_demand_diff_wt_gdow_q10','cluster_prob_gdowh','demand_rolling_4',
            'dist_to_center_km'
        ]
        self.attributed_by = attributed_by
        self.target_column = target_column
        self.df_train = None
        self.preprocess = None
        self.mdl = None
    
    def make_model(X):
        raise Exception('Not Implemented')
        return None
    
    def fit(self, X, y, target_preprocessor=None, *args, **kwargs):
        self.make_model(X)   
        y_target = y
        
        if target_preprocessor:
            y_target = target_preprocessor.fit_transform(y)
            
        self.mdl.fit(X, y_target, *args, **kwargs)
        return self
    
    def fit_raw(self, df_train, target_column='demand', force_redo_preproces=False, target_preprocessor=None, *args, **kwargs):
        """
        Predict with raw data, as given from the competition
        
        raw_data attributes:
            geohash6
            day
            timestamp
            demand
        """
        self.target_preprocessor = target_preprocessor
        
        if (self.df_train is None or force_redo_preproces):
            self.df_train = preprocess(df_train)
            self.df_train = more_preprocess(self.df_train)
            
        X = self.df_train[self.categorical_features + self.numerical_features]
        y = self.df_train[self.target_column]
        return self.fit(X,y, *args, **kwargs)
    
    def predict(self, X):
        return self.mdl.predict(X)
    
    def predict_futures(self, attributions=None, n_ticks=5):
        """
        Predict the futures from n_ticks beyond training data
            
        """
        
        # get the last timestamp in train data
        # and expand_ticks time from it
        temp_features = ['day', 'hour', 'minute']
        temp = self.df_train.groupby('geohash6')[temp_features + ['demand']].apply(
            lambda x: x.sort_values(by=temp_features, ascending=False).head(1)
        )
        future_rows=[]
        temp_arr = temp.reset_index().set_index('geohash6')[temp_features].apply(
            lambda x: get_future_n_ticks(*x,n_ticks), axis=1
        )
        temp_x = temp_arr.reset_index().apply(
            lambda r: [future_rows.append([r.geohash6] + list(t)) for t in r[0]], axis=1
        )
        # remove unused variables
        del temp_x
        future_df = pd.DataFrame(
            future_rows,
            columns=['geohash6', 'day', 'hour', 'minute']
        )
        
        if attributions is None:
            attributions = self.attributions

        x = preprocess(future_df)
        x = x.merge(
            self.df_train[
                self.categorical_features + self.numerical_features
            ].groupby(self.attributed_by).agg('mean').reset_index(),
            left_on=self.attributed_by,
            right_on=self.attributed_by,
            suffixes=('', '_r'),
            how='left'
        )
        future_df['y_hat'] = self.predict(x)
        return future_df


class MyXGBModel(RegressorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def make_model(self, X):
        self.preprocess = make_column_transformer(
            (self.categorical_features, SimpleImputer()),
            (self.numerical_features, SimpleImputer(strategy='constant', fill_value=0))
        )
        self.mdl = make_pipeline(
            self.preprocess,
            XGBRegressor(
                n_jobs=4,
                n_estimators=100
            )
        )


class MyExtraTreeModel(RegressorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def make_model(self, X):
        self.preprocess = make_column_transformer(
            (self.categorical_features, SimpleImputer()),
            (self.numerical_features, SimpleImputer(strategy='constant', fill_value=0))
        )
        self.mdl = make_pipeline(
            self.preprocess,
            ExtraTreesRegressor(
                n_jobs=4,
                n_estimators=100
            )
        )


class MyCatBoostModel(RegressorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def make_model(self, X):
        """
        After a few attempts, I've found that loss function in Catboost does not reduced significantly after 
        100 iterations, so let's stop at this local minima
        """
        self.category_indexes = [
            X.columns.get_loc(f) for f in self.categorical_features
        ]
        self.preprocess = make_column_transformer(
            (self.categorical_features, SimpleImputer()),
            (self.numerical_features, SimpleImputer(strategy='constant', fill_value=0))
        )
        self.mdl = make_pipeline(
            self.preprocess,
            CatBoostRegressor(
                cat_features=self.category_indexes, 
                iterations=100,
                early_stopping_rounds=100)
        )
        
        
class MyElasticNetModel(RegressorModel):
    """
    Fit An ElasticNet Model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def make_model(self, X):
        self.category_indexes = [
            X.columns.get_loc(f) for f in self.categorical_features
        ]
        self.preprocess = make_column_transformer(
            # for linear models, need to change categorical features into one hot encoding
            (self.categorical_features, OneHotEncoder()),
            (self.numerical_features, SimpleImputer(strategy='constant', fill_value=0))
        )
        self.mdl = make_pipeline(
            self.preprocess,
            ElasticNet()
        )
        

class MyDL1Model(RegressorModel):
    """
    Fit custom MLP
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def make_dl_model(self, input_dim):
        mdl = Sequential()
        mdl.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
        mdl.add(Dense(256, activation='relu'))
        mdl.add(Dropout(0.2))
        mdl.add(Dense(64, activation='relu'))
        mdl.add(Dropout(0.2))
        mdl.add(Dense(8, activation='relu'))
        mdl.add(Dense(1, activation='linear'))
        
        mdl.compile(
            optimizer=Adam(lr=0.002, clipnorm=1., decay=0.001),
            loss='mape',
            metrics=['mse', 'mae', 'mape']
        )
        
        print(mdl.summary())
        return mdl
    
    def get_input_dim(self):
        input_dim = 0
        for cat_feature in self.categorical_features:
            input_dim += self.df_train[cat_feature].unique().shape[0]
        input_dim += len(self.numerical_features)
        return input_dim
        
        
    def make_model(self, X):
        self.preprocess = make_column_transformer(
            # for deep learning models, need to change categorical features into one hot encoding
            (self.categorical_features, OneHotEncoder()),
            (self.numerical_features, SimpleImputer(strategy='constant', fill_value=0))
        )
        self.mdl = make_pipeline(
            self.preprocess,
            self.make_dl_model(self.get_input_dim())
        )
        


class MySimple2LvXGBoostModel(RegressorModel):
    """
    Ensembling the model to reduce 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.mdl_lv2 = None
        
    def make_model(self, X):
        self.mdl = MyXGBModel()
        self.mdl_lv2 = MyXGBModel()
        
    def fit_raw(self, df_train, target_column='demand', force_redo_preproces=False, target_preprocessor=None, *args, **kwargs):
        self.make_model(None)
        features = self.mdl.categorical_features + self.mdl.numerical_features
        
        self.mdl.fit_raw(
            df_train, 
            target_column=target_column, 
            force_redo_preproces=force_redo_preproces,
            target_preprocessor=target_preprocessor,
            *args,
            **kwargs
        )
        
        predictions = self.mdl.predict(self.mdl.df_train[features])
        resid = self.mdl.df_train[target_column] - predictions
        
        base_columns = [
            'geohash6',
            'day',
            'timestamp',
            'demand'
        ]
        lv2_df_train = self.mdl.df_train[base_columns].copy()
        lv2_df_train['demand'] = resid
        
        self.mdl_lv2.fit_raw(lv2_df_train)
        
    def predict_futures(self, attributions=None, n_ticks=5):
        lv1_predictions = self.mdl.predict_futures(attributions=attributions, n_ticks=n_ticks)
        lv2_predictions = self.mdl_lv2.predict_futures(attributions=attributions, n_ticks=n_ticks)
        
        lv1_predictions['lv1_yhat'] = lv1_predictions['y_hat']
        lv2_predictions['lv2_yhat'] = lv2_predictions['y_hat']
        
        join_on = [
            'geohash6',
            'day',
            'hour',
            'minute',
            'is_weekend',
            'dow',
            'lat',
            'long',
            'hour_p8',
            'is_working_hour'
        ]
        final_predictions = lv1_predictions.merge(
            lv2_predictions,
            how='inner',
            left_on=join_on,
            right_on=join_on
        )
        
        final_predictions['y_hat'] = final_predictions['lv1_yhat'] + final_predictions['lv2_yhat']
    
        return final_predictions
        
