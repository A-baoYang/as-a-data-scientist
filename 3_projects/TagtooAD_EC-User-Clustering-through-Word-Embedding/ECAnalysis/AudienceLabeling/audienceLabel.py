import argparse
import numpy as np
import pandas as pd
import dask.dataframe as dd
import datetime, time, csv
import os
import sys
sys.path.append(os.path.abspath('../'))   # go to custom function root folder
import EC_AllReportFunctions as ec   # import ECReport functions
import upload_GoogleStorage as upload_gs   # import google storage functions

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=int, required=True, default=191225,
                        help='The semaphore counter of asyncio.')
    parser.add_argument('--end_date', type=int, required=True, default=191231,
                        help='Number of connection for aiohttp client session.')
    parser.add_argument('--ec_id', type=int, required=False, default=1039,
                        help='Number of connection for aiohttp client session.')
    args = parser.parse_args()
    return args

#import EC_AllReportFunctions as ec 
#使用到的裡面的function
'''

def findMedian(df, group_col, S):
    s = 0
    tmp = 0
    for i, p in zip(df[str(group_col)], df['percent']):
        tmp += p
        if tmp > int(S):
            s = i
            break
    return s

def freqTable(df_inter):
    oH = pd.get_dummies(df_inter[['type']])
    oH = pd.concat([df_inter[['track_user']], oH], axis=1)
    t = oH.groupby(['track_user'])['type_track', 'type_transaction'].sum().reset_index()
    t = t.rename(columns={'type_transaction': 'trans_times'})
    t['track_times'] = t['type_track'] + t['trans_times']
    t = t.drop('type_track', axis=1)
    return t

def reverseFreqTable(t, group_col, count_col):
    d = t.groupby([str(group_col)])[str(count_col)].count().reset_index().sort_values(str(group_col), ascending=False)
    d['percent'] = d[str(count_col)] / d[str(count_col)].sum() * 100
    return d

def labelUserGroup(df, col, FU, FP, NFU, NFP, VIP):
    col = col
    conditions = [
        df[col].isin(set(FU)&set(FP)),
        df[col].isin(set(NFU)&set(FP)),
        df[col].isin(set(FU)&set(NFP)),
        df[col].isin(set(NFU)&set(NFP))
    ]
    choices = [0, 1, 2, 3]  # 0: 主力組  1: 潛力組  2: 猶豫組  3: 路人組
    df['userGroup'] = np.select(conditions, choices, default=-1)
    df['vip'] = np.where(df[col].isin(VIP), 1, 0) # 0: 非VIP  1: VIP
    df = df[df['userGroup'] != -1]
    return df

'''


# label users every week (temporarily)
args = parse_arguments()
start_date = args.start_date
end_date = args.end_date
ec_id = args.ec_id
dataUrl = f'gs://tagtoo-bigquery-export/ECAnalysis/{ec_id}_trackTrans_{start_date}_{end_date}/*'
print('fetching user records from Storage...')
df = ec.queryFromBQ(dataUrl)


# google storage auth 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/jovyan/.config/gcloud/application_default_credentials.json"
bq_project = 'gothic-province-823'


# label past week users 
print('labeling users...')
t = ec.freqTable(df)
vt = df.groupby(['track_user'])['value'].sum().reset_index().sort_values('value', ascending=False)
vt = vt[vt['value'] > 0]
U_rule = ec.findMedian(ec.reverseFreqTable(t, 'track_times', 'track_user'), 'track_times', 50)
P_rule = ec.findMedian(ec.reverseFreqTable(t, 'trans_times', 'track_user'), 'trans_times', 50)
VIP_rule = ec.findMedian(ec.reverseFreqTable(vt, 'value', 'track_user'), 'value', 20)
FU = t[(t['track_times'] > U_rule)].track_user.values
NFU = t[(t['track_times'] <= U_rule)].track_user.values
FP = t[(t['trans_times'] > P_rule)].track_user.values
NFP = t[(t['trans_times'] <= P_rule)].track_user.values
VIP = vt[(vt['value'] > VIP_rule)].track_user.values

print('50%的用戶瀏覽 ', U_rule, '次 / 50%的用戶購買 ', P_rule, '次 / 在所有購買者中，前20%高消費族群至少每人花了 ', VIP_rule, '元')
print('totalUserNum: ', len(t), ' . FU: ', len(FU), ' / NFU: ', len(NFU), ' / FP: ', len(FP), ' / NFP: ', len(NFP), ' / VIP: ', len(VIP))
print('0 主力組: ', len(set(FU)&set(FP)), ' / 1 潛力組: ', len(set(NFU)&set(FP)), ' / 2 猶豫組 : ', len(set(FU)&set(NFP)), ' / 3 路人組: ', len(set(NFU)&set(NFP)), ' . TotalNum: ', len(set(FU)&set(FP))+len(set(FU)&set(NFP))+len(set(NFU)&set(FP))+len(set(NFU)&set(NFP)))

df_labeled = ec.labelUserGroup(df, 'track_user', FU, FP, NFU, NFP, VIP)
df_labeled = df_labeled[['track_user', 'advertiser_id', 'userGroup', 'vip']]


# upload to google storage
print('uploading to google storage tagtoo bucket...')
filename = f'{ec_id}_audience_label_{start_date}_{end_date}.csv'
upload_gs.upload_StringIO(df_labeled, bq_project, 'tagtoo-bigquery-export', f'ECAnalysis/audienceLabel/{filename}')

