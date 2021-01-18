import os
import logging
import argparse
import time, csv
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import dask.dataframe as dd
import datetime as dt
import upload_GoogleStorage as upload_gs

from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from utils import (
    queryFromBQ,
    unique_user_by_shift
)

## google storage auth 
# GSIO = google_storage.GoogleStorageIO()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "application_default_credentials.json"
bq_project = 'gothic-province-823'
bq_bucket = 'tagtoo-bigquery-export'
base_folder ='pureEC_timeShifting_validation'
gcs_base_folder = f'ECAnalysis/{base_folder}'

OUTPUTS = {
    'output': 'output.txt'
}


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_date', type=str, required=True, 
                        help='Start date of training set.')
    parser.add_argument('--to_date', type=str, required=True, 
                        help='End date of training set.')
    parser.add_argument('--predict_month', type=int, required=True,
                        help='The month of user label to predict.')
    parser.add_argument('--predict_year', type=int, required=True,
                        help='The year of user label to predict.')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Threshold for Logistic Regression proba classify.')
    args = parser.parse_args()
    return args


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    
    # variables
    predict_month = args.predict_month
    threshold = args.threshold
    from_date = args.from_date
    to_date = args.to_date
    gcs_output_folder = f'{gcs_base_folder}/ulabel_modelResults_{from_date}_{to_date}'
    
    print('Read train data from GCS...')
    
    ## import labeled data
    gcs_ulabel_folder = f'{gcs_base_folder}/ulabel_{from_date}_{to_date}'
    ulabel_MBS2WO_shifts = list()
    for i in range(0,10):  # here can revise to use * in link 
        labeled_data_path = f'gs://{bq_bucket}/{gcs_ulabel_folder}/ulabel_from_month_{predict_month-2}_MBS2WO_shift_{i}.csv'
        ulabel_MBS2WO_shifts.append(queryFromBQ(labeled_data_path))
    
    ## import labeled validation dataset
    labeled_val_data_path = f'gs://{bq_bucket}/{gcs_ulabel_folder}/val__month_{predict_month-1}.csv'
    ulabel_val = queryFromBQ(labeled_val_data_path)
    
    ## import buyer info data
    buyer_info_path = f'gs://{bq_bucket}/{gcs_ulabel_folder}/buyer_purchase_info_cross_ec_ind_by_shifts.csv'
    buyer_info_dummy = queryFromBQ(buyer_info_path)
    
    
    ## 紀錄各個切點 => 因為有的label包出來的user數太少 改1個切點(兩組)
    label_cols = ['Avg_Item_Value','Avg_Order_Value']
    label_items = {
        label_cols[0]: ulabel_MBS2WO_shifts[0][label_cols[0]].unique(),
        label_cols[1]: ulabel_MBS2WO_shifts[0][label_cols[1]].unique()
    }
    
    
    lr_coefs = list()
    for i in range(0,2):
        for label_item in label_items[label_cols[i]]:
            print(f'==={label_cols[i]}: {label_item}===')
            userlists, specific_label_all_user = unique_user_by_shift(ulabel_MBS2WO_shifts, i, label_cols, label_item)
            val_userlist_tmp_ = ulabel_val[ulabel_val[label_cols[i]] == label_item]['track_user'].unique()

            df_check = pd.DataFrame({'track_user': specific_label_all_user})
            df_check = pd.merge(df_check, buyer_info_dummy, on='track_user', how='left')

            for s in range(0, 10):
                df_check[f'shift_{s}'] = np.where(df_check['track_user'].isin(userlists[s]), 1, 0)
            df_check['val'] = np.where(df_check['track_user'].isin(val_userlist_tmp_), 1, 0)

            class_weight = int((len(df_check) - df_check['val'].sum()) / df_check['val'].sum())
            print('class_weight: ', class_weight)

            X = df_check.iloc[:, 1:df_check.shape[1]-1]
            Y = df_check.iloc[:, df_check.shape[1]-1]
            train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
            # define class weights
            w = {0:1, 1:class_weight}
            lr = LogisticRegression(random_state=0, class_weight=w).fit(train_x, train_y)

            adjusted_predict = list()
            for a in lr.predict_proba(test_x)[:,1]:
                adjusted_predict.append(1) if a >= threshold else adjusted_predict.append(0)

            print('training set accuracy: ', lr.score(train_x, train_y))
            print('testing set accuracy: ', lr.score(test_x, test_y))
            lr_coefs.append(lr.coef_)
            print(confusion_matrix(test_y, adjusted_predict))
            print(classification_report(test_y, adjusted_predict))
            

    df_coefs = pd.DataFrame([list(x[0]) for x in lr_coefs], columns=df_check.iloc[:,1:df_check.shape[1]-1].columns)

    coefs_path = f'pureEC_Purchase_Power__train__LR_coefs.csv'
    coefs_gcs_path = f'{gcs_output_folder}/{coefs_path}'
    df_coefs.to_csv(coefs_path, index=False, encoding='utf-8')
    
    upload_gs.upload_StringIO(coefs_path, bq_project, bq_bucket, coefs_gcs_path)
#     GSIO.upload_file(gsuri=coefs_gcs_path, localpath=coefs_path)

    with open(OUTPUTS['output'], 'w+') as f:
        f.write(str(coefs_gcs_path))

    print('Job complete.')
    
    
if __name__ == '__main__':
    main()


