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
    'output': 'output_predict.txt'
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
    predict_year = args.predict_year
    threshold = args.threshold
    from_date = args.from_date
    to_date = args.to_date
    gcs_output_folder = f'{gcs_base_folder}/ulabel_modelResults_{from_date}_{to_date}'
    
    print('Read train data from GCS...')
    
    ## import testset
    gcs_ulabel_folder = f'{gcs_base_folder}/ulabel_{from_date}_{to_date}'
    ulabel_MBS2WO_shifts = list()
    for i in range(0,10):  # here can revise to use * in link 
        data_path = f'gs://{bq_bucket}/{gcs_ulabel_folder}/ulabel_from_month_{predict_month-1}_MBS2WO_shift_fp_{i}.csv'
        ulabel_MBS2WO_shifts.append(queryFromBQ(data_path))
    
    label_cols = ['Avg_Item_Value','Avg_Order_Value']
    label_items = {
        label_cols[0]: ulabel_MBS2WO_shifts[0][label_cols[0]].unique(),
        label_cols[1]: ulabel_MBS2WO_shifts[0][label_cols[1]].unique()
    }
    
    ## import buyer info data
    buyer_info_path = f'gs://{bq_bucket}/{gcs_ulabel_folder}/buyer_purchase_info_cross_ec_ind_by_shifts_fp.csv'
    buyer_info_dummy = queryFromBQ(buyer_info_path)
    
    ## import coefs
    coefs_path = f'gs://{bq_bucket}/{gcs_output_folder}/pureEC_Purchase_Power__train__LR_coefs.csv'
    df_coefs = queryFromBQ(coefs_path)
    
    
    ## predict
    count = 0
    for i in range(0,2):
        for label_item in label_items[label_cols[i]]:
            print(f'==={label_cols[i]}: {label_item}===')
            userlists, specific_label_all_user = unique_user_by_shift(ulabel_MBS2WO_shifts, i, label_cols, label_item)
            print('specific_label_all_user: ', len(specific_label_all_user))

            df_check = pd.DataFrame({
                'track_user': specific_label_all_user
            })
            df_check = pd.merge(df_check, buyer_info_dummy, on='track_user', how='left')

            for s in range(0, 10):
                df_check[f'shift_{s}'] = np.where(df_check['track_user'].isin(userlists[s]), 1, 0)
            print('df_check shape: ', df_check.shape)
            
            df_coefs.drop([nd for nd in df_coefs.columns if nd not in df_check.iloc[:,1:df_check.shape[1]-1].columns],axis=1,inplace=True)
            df_check.drop([nd for nd in df_check.iloc[:,1:df_check.shape[1]-1].columns if nd not in df_coefs.columns],axis=1,inplace=True)

            X = df_check.iloc[:, 1:df_check.shape[1]-1].values
            sum_product = list()
            threshold = args.threshold
            for row in tqdm(range(0, len(df_check))):
                sum_product.append((X[row] * df_coefs.iloc[count, :].values).sum())
            df_check['origin_score'] = sum_product
            count += 1

            for row in tqdm(range(0, len(X))):
                if sum_product[row] >= threshold:
                    sum_product[row] = 1
                else:
                    sum_product[row] = 0

            df_check['adjust_val'] = sum_product
            result_path = f'prediction__{predict_year}_{predict_month}__{label_cols[i]}_{label_item}.csv'
            df_check.to_csv(result_path, index=False, encoding='utf-8')
            result_gcs_path = f'{gcs_output_folder}/{result_path}'
            
            upload_gs.upload_StringIO(result_path, bq_project, bq_bucket, result_gcs_path)
#             GSIO.upload_file(gsuri=result_gcs_path, localpath=result_path)

            with open(OUTPUTS['output'], 'a+') as f:
                f.write(str(result_gcs_path))

    print('Job complete.')

    
if __name__ == '__main__':
    main()



