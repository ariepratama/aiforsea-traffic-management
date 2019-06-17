import pandas as pd
import numpy as np
import argparse
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path')
    parser.add_argument('--prediction-future-ticks', help='1 ticks is 15 minutes for all geo', default=5)
    parser.add_argument('--training-start-from-day', help='start training from which day', default=1)
    parser.add_argument('--training-day-length', help='how many days should the training consists of', default=14)
    parser.add_argument('--prediction-result-path', default='result.csv')
    args = parser.parse_args()

    print('Arguments: {}'.format(args))
    train_data_path = args.train_data_path
    file_format = train_data_path.split('.')[-1]
    if file_format == 'zip':
        df_train = pd.read_csv(train_data_path, compression=file_format)
    else:
        df_train = pd.read_csv(train_data_path)

    df_train = get_train_data(
        df_train, 
        start_day=args.training_start_from_day, 
        n_days=args.training_day_length
    )
    df_train = fill_missing_timeframe(df_train)
    
    print('Training data shape: {}'.format(df_train.shape))
    
    mdl = MyXGBModel()
    print('Training the model...')
    mdl.fit_raw(df_train)

    print('Writing result...')
    future_predictions = mdl.predict_futures(n_ticks=args.prediction_future_ticks)
    future_predictions.to_csv(args.prediction_result_path)

    print('Finished processing ! Result at: {}'.format(args.prediction_result_path))

