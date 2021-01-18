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
from calendar import monthrange
from utils import (
    queryFromBQ,
    page_preprocessing,
    value_preprocessing,
    gen_pureEC_dataframe,
    gen_user_label,
    generate_shifts,
    generate_buyer_info_dummy
)

## google storage auth 
# GSIO = google_storage.GoogleStorageIO()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "application_default_credentials.json"
bq_project = 'gothic-province-823'
bq_bucket = 'tagtoo-bigquery-export'
base_folder ='pureEC_timeShifting_validation'
gcs_base_folder = f'ECAnalysis/{base_folder}'

OUTPUTS = {
    'labeled_data': 'labeled_data.txt',
    'labeled_val_data': 'labeled_val_data.txt',
    'for_predict': 'for_predict.txt',
    'buyer_ec_ind_info': 'buyer_ec_ind_info.txt',
    'buyer_ec_ind_info_for_predict': 'buyer_ec_ind_info_for_predict.txt'
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
    parser.add_argument('--itemValueCut', type=int, required=True,
                        help='The cutting point of Avg. Item Value in user label.')
    parser.add_argument('--orderValueCut', type=int, required=True,
                        help='The cutting point of Avg. Order Value in user label.')
    args = parser.parse_args()
    print(args)
    return args


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    
    ## variables
    from_date = args.from_date
    to_date = args.to_date
    dataUrl = f'gs://{bq_bucket}/{gcs_base_folder}/1604visitor_in_pureEC_{from_date}_{to_date}/*'  # if user agrs will get error
    gcs_output_folder = f'{gcs_base_folder}/ulabel_{from_date}_{to_date}'
    predict_month = args.predict_month
    v_Q1 = args.itemValueCut
    aov_Q1 = args.orderValueCut
    
    print('Loading Data...')
    df = queryFromBQ(dataUrl)
    
    print('Preprocessing Data...')
    df = page_preprocessing(df)
    
    ## Add "Average Product Value (APV)"
    df['APV'] = df['value'] / df['num_items']
    df_twd = df[(df['currency'] == 'TWD') | (df['currency'] == 'NTD')]
    df_ovs = df[(df['currency'] != 'TWD') & (df['currency'] != 'NTD')]
    ## value preprocessing
    df_twd = value_preprocessing(df_twd)
    df_twd_pureEC = gen_pureEC_dataframe(df_twd)
    ## custom 處理 加年份
    df_twd_pureEC['datetime'] = df_twd_pureEC['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_twd_pureEC['year'] = df_twd_pureEC['pageview_time'].apply(lambda x: int(x.split(' ')[0].split('-')[0]))
    ## clean the date-range from BigQuery
    df_twd_pureEC = df_twd_pureEC[~((df_twd_pureEC['year']==2020)&(df_twd_pureEC['view_bymonth']==predict_month+1))]
    ## generate train & test set
    df_test = df_twd_pureEC[(df_twd_pureEC['year']==2020)&(df_twd_pureEC['view_bymonth']==predict_month)]
    df_twd_pureEC = df_twd_pureEC[~((df_twd_pureEC['year']==2020)&(df_twd_pureEC['view_bymonth']==predict_month))]

    
    ## generate val set
    val = df_twd_pureEC[df_twd_pureEC['view_bymonth']==predict_month-1]
    ulabel_val = gen_user_label(val, shiftType='validation', 
                                months=1, group_num=2, v_Q1 = v_Q1, v_Q2 = None, v_Q3 = None, 
                                aov_Q1 = aov_Q1, aov_Q2 = None, aov_Q3 = None, f_Q1=1, f_Q2=None, f_Q3=None)
    labeled_val_data_path = f'val__month_{predict_month-1}.csv'
    labeled_val_data_gcs_path = f'{gcs_output_folder}/{labeled_val_data_path}'
    ulabel_val['user_label'].to_csv(labeled_val_data_path, index=False, encoding='utf-8')
    
    upload_gs.upload_StringIO(labeled_val_data_path, bq_project, bq_bucket, labeled_val_data_gcs_path)
#     GSIO.upload_file(gsuri=labeled_val_data_gcs_path, localpath=labeled_val_data_path)

    with open(OUTPUTS['labeled_val_data'], 'w+') as f:
        f.write(str(labeled_val_data_gcs_path))
        

    ## generate shifts
    rounds_list, shift_start, shift_start_timing, shift_end_timing = generate_shifts(df_twd_pureEC, predict_month, for_predict=False)
    rounds_list_fp, shift_start_fp, shift_start_timing_fp, shift_end_timing_fp = generate_shifts(df_twd_pureEC, predict_month, for_predict=True)
    

    ## count user label from these shifts
    # ulabel_MBS2WO_shifts = list()
    for _count, frame in tqdm(enumerate(rounds_list)):
        ulabel_MBS2WO_shift_ = gen_user_label(frame, shiftType='month_base_shift_double-weekly_overlap', 
                                                       months=1, group_num=2,
                                                       v_Q1 = 500, v_Q2 = None, v_Q3 = None, 
                                                       aov_Q1 = 1000, aov_Q2 = None, aov_Q3 = None,
                                                       f_Q1=1, f_Q2=None, f_Q3=None)
    #     ulabel_MBS2WO_shifts.append(ulabel_MBS2WO_shift_)
        labeled_data_path = f'ulabel_from_month_{predict_month-2}_MBS2WO_shift_{_count}.csv'
        labeled_data_gcs_path = f'{gcs_output_folder}/{labeled_data_path}'
        ulabel_MBS2WO_shift_['user_label'].to_csv(labeled_data_path, index=False, encoding='utf-8')

        upload_gs.upload_StringIO(labeled_data_path, bq_project, bq_bucket, labeled_data_gcs_path)
#         GSIO.upload_file(gsuri=labeled_data_gcs_path, localpath=labeled_data_path)

        with open(OUTPUTS['labeled_data'], 'w+') as f:
            f.write(str(labeled_data_gcs_path))
            
    
    for _count, frame in tqdm(enumerate(rounds_list_fp)):
        ulabel_MBS2WO_shift_ = gen_user_label(frame, shiftType='month_base_shift_double-weekly_overlap', 
                                                       months=1, group_num=2,
                                                       v_Q1 = 500, v_Q2 = None, v_Q3 = None, 
                                                       aov_Q1 = 1000, aov_Q2 = None, aov_Q3 = None,
                                                       f_Q1=1, f_Q2=None, f_Q3=None)
        labeled_data_path = f'ulabel_from_month_{predict_month-1}_MBS2WO_shift_fp_{_count}.csv'
        labeled_data_gcs_path = f'{gcs_output_folder}/{labeled_data_path}'
        ulabel_MBS2WO_shift_['user_label'].to_csv(labeled_data_path, index=False, encoding='utf-8')
        
        upload_gs.upload_StringIO(labeled_data_path, bq_project, bq_bucket, labeled_data_gcs_path)
#         GSIO.upload_file(gsuri=labeled_data_gcs_path, localpath=labeled_data_path)

        with open(OUTPUTS['for_predict'], 'w+') as f:
            f.write(str(labeled_data_gcs_path))


    buyer_info_dummy = generate_buyer_info_dummy(df_twd_pureEC, shift_start, shift_start_timing, shift_end_timing)
    buyer_info_path = 'buyer_purchase_info_cross_ec_ind_by_shifts.csv'
    buyer_info_dummy.to_csv(buyer_info_path, index=False, encoding='utf-8')
    buyer_info_gcs_path = f'{gcs_output_folder}/{buyer_info_path}'
    
    upload_gs.upload_StringIO(buyer_info_path, bq_project, bq_bucket, buyer_info_gcs_path)
#     GSIO.upload_file(gsuri=buyer_info_gcs_path, localpath=buyer_info_path)

    with open(OUTPUTS['buyer_ec_ind_info'], 'w+') as f:
        f.write(str(buyer_info_gcs_path))
        
        
    buyer_info_dummy_fp = generate_buyer_info_dummy(df_twd_pureEC, shift_start_fp, shift_start_timing_fp, shift_end_timing_fp)
    buyer_info_path = 'buyer_purchase_info_cross_ec_ind_by_shifts_fp.csv'
    buyer_info_dummy.to_csv(buyer_info_path, index=False, encoding='utf-8')
    buyer_info_gcs_path = f'{gcs_output_folder}/{buyer_info_path}'
    
    upload_gs.upload_StringIO(buyer_info_path, bq_project, bq_bucket, buyer_info_gcs_path)
#     GSIO.upload_file(gsuri=buyer_info_gcs_path, localpath=buyer_info_path)

    with open(OUTPUTS['buyer_ec_ind_info_for_predict'], 'w+') as f:
        f.write(str(buyer_info_gcs_path))

    print('Job complete.')
    
    
if __name__ == '__main__':
    main()


