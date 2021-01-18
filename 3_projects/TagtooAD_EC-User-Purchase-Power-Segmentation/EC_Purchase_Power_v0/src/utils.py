import os
import pytz
tw = pytz.timezone('Asia/Taipei')
import numpy as np
import pandas as pd
import dask.dataframe as dd
import time, csv
import datetime as dt

from tqdm import tqdm
from datetime import datetime
from calendar import monthrange


def queryFromBQ(bigquery_goal):
    """
    fetch data from BigQuery Bucket
    
    Parameters
    ----------
    bigquery_goal :
    bigquery uri
    
    Returns
    ----------
    The corresponding dataframe table at Bigquery bucket
    
    """
    s = time.time()
    t = dd.read_csv(bigquery_goal, dtype={'value': 'float64'}, encoding='utf-8')
    df = t.compute()
    print('data length: ', len(df))
    if 'track_user' in df.columns:
        print('#UU: ', len(df.track_user.unique()))
    print(f'Loading Time = {time.time()-s}')
    return df


def page_preprocessing(df):
    # 缺漏值處理
    df['page_key'] = df['page_key'].fillna('None')
    df['page'] = df['page'].fillna('None')
    df['content_ids'] = df['content_ids'].fillna('missing')
    
    # type 誤值處理
    df['type'] = np.where((df['type'].str.contains('track') & df['page_key'].str.contains('trans')), 'transaction', df['type'])
    df['type'] = np.where((df['type'].str.contains('track') & df['page'].str.contains('Complete|complete|Finish|finish|Order|order|Payment|payment|Thank|thank|Success|success|sucess|oid=|Trans|trans|Result|result|Purchase|purchase|deal|Deal')), 'transaction', df['type'])

    df = df[df['type'] == 'transaction']
    
    ## ind=5 lumba.id currency revise
    df['currency'] = np.where(df['advertiser_id'] == 1590, 'IDR', df['currency'])
    
    ## ecid=100 的計價單位是萬
    df['value'] = np.where(df['advertiser_id'] == 100, df['value']*10000, df['value'])
    
    return df


def value_preprocessing(df_twd):
    # APV 平均單品價格 > 10000，但轉換動作不是購買： 100(中信看房), 1305(信義看房), 387(hotcar看車), 1466(旅遊咖需求預約), 1168(山富旅遊), 1499(食尚玩家看文章)
    # value 數字重複，很高機率是送事件時候記錯的 EC：1167(>11130), 844, 1345, 95, 153, 1011, 1221
    twd_housing = [100,1305]
    twd_car = [387]
    twd_travel = [1168,1466]
    twd_media = [1499]
    ec_wrong_price = [95,153,844,1011,1221,1345]

    ## 修正 APV 小於1 的紀錄
    df_twd['num_items'] = np.where(df_twd['APV'] < 1, df_twd['content_ids'].apply(lambda x: len(x.split(','))), df_twd['num_items'])
    # 更新 APV
    df_twd['APV'] = df_twd['value'] / df_twd['num_items']
    # 還是小於1的讓他等於1
    df_twd['value'] = np.where(df_twd['APV'] < 1, df_twd['num_items'], df_twd['value'])
    # 再次更新 APV
    df_twd['APV'] = df_twd['value'] / df_twd['num_items']

    ## 去除不要用來算 EC 平均單品價的紀錄
    df_twd = df_twd.reset_index()
    df_twd = df_twd.reset_index()
    df_twd.drop('index', axis=1, inplace=True)

    remv1 = df_twd[df_twd['advertiser_id'].isin(twd_housing+twd_car+twd_travel+twd_media)]['level_0'].values
    remv2 = df_twd[(df_twd['APV'] > 10000) & (df_twd['advertiser_id'].isin(ec_wrong_price))]['level_0'].values
    remv3 = df_twd[(df_twd['APV'] > 11300) & (df_twd['advertiser_id'] == 1167)]['level_0'].values
    df_t = df_twd[~df_twd['level_0'].isin(list(remv1)+list(remv2)+list(remv3))]

    ## 計算各間 EC 的平均單品價格
    has_value = df_t[~df_t['value'].isnull()]
    value_per_item = has_value.groupby(['advertiser_id'])['value'].sum() / has_value.groupby(['advertiser_id'])['num_items'].sum()
    value_per_item = value_per_item.reset_index()
    value_per_item.rename(columns={0: 'value_per_item'}, inplace=True)

    ## 用各間 EC 平均單品價格 乘以 該紀錄的 num_items 補缺漏的 value值
    df_twd['value'] = df_twd['value'].mask(df_twd['value'].isnull(), df_twd['advertiser_id'].map(value_per_item.set_index('advertiser_id')['value_per_item']) * df_twd['num_items'])
    
    ## 也把記錯的 value 用同 EC 平均值修正
    df_twd['value'] = df_twd['value'].mask(df_twd['advertiser_id'].isin(ec_wrong_price+[1167]), df_twd['advertiser_id'].map(value_per_item.set_index('advertiser_id')['value_per_item']) * df_twd['num_items'])

    ## 計算各產業的平均單品價格
    value_per_item_IND = has_value.groupby(['industry_id'])['value'].sum() / has_value.groupby(['industry_id'])['num_items'].sum()
    value_per_item_IND = value_per_item_IND.reset_index()
    value_per_item_IND.rename(columns={0: 'value_per_item'}, inplace=True)

    ## 沒有其他現存 value 可以參考的 ECs，用產業平均單品價格來補
    # [ 765(OKtea),  927(Levi's),  824(愛車褓母 車用品), 1021(Qmomo 服飾),  981(Joanna Shop 服飾), 1041(王德傳茶莊),  997(韓秀姬),  875(darling pet),  585(Paktor 升級內購), 1062(JREP Clothing),  368(colorfulmarine), 291(vitalspa), 1299(Qmomo(馬來西亞))]
    df_twd['value'] = df_twd['value'].mask(df_twd['value'].isnull(), df_twd['industry_id'].map(value_per_item_IND.set_index('industry_id')['value_per_item']) * df_twd['num_items'])

    ## 最後 ind=18 因為只有 585(Paktor) 的資料，無法參照；詳細看過是預約轉換，value = 1
    df_twd['value'] = df_twd['value'].fillna(1.0)
    
    # 最後更新 APV
    df_twd['APV'] = df_twd['value'] / df_twd['num_items']
    return df_twd


def gen_pureEC_dataframe(df_twd):
    # 轉換事件非購買的 EC （房產、汽車、旅遊、投保理財、內容、其他預約動作）
    value_per_item_NEW = df_twd.groupby(['advertiser_id'])['value'].sum() / df_twd.groupby(['advertiser_id'])['num_items'].sum()
    value_per_item_NEW = value_per_item_NEW.reset_index()
    value_per_item_NEW.rename(columns={0: 'value_per_item'}, inplace=True)
    ec_reserve = value_per_item_NEW[value_per_item_NEW['value_per_item'] < 50].advertiser_id.unique()
    ec_pure = value_per_item_NEW[value_per_item_NEW['value_per_item'] >= 50].advertiser_id.unique()

    twd_housing = [100,708,787,1201,1202,1204,1305,1497,1538,1555]
    twd_car = [155,292,387,824,1213]
    twd_insurFin = [1557]
    twd_travel = [1168,1245]
    twd_media = [1043]
    twd_service = [252,793,1589,1599] # socie, fersonal, Eng4U, VoiceTube
    twd_social = [585]
    twd_other_reserve = [782,1227,1284,1436,1460,1508,1555,1583,1584,1602]

    twd_pureEC = value_per_item_NEW[(~value_per_item_NEW['advertiser_id'].isin(twd_housing+twd_car)) & (value_per_item_NEW['value_per_item'] >= 50)].advertiser_id.values
    df_twd_pureEC = df_twd[df_twd['advertiser_id'].isin(twd_pureEC)]
    return df_twd_pureEC


def gen_user_label(df_twd_pureEC, shiftType, months, group_num, v_Q1, v_Q2, v_Q3, aov_Q1, aov_Q2, aov_Q3, f_Q1, f_Q2, f_Q3):
    if group_num == 4:
        # 平均每個商品單價
        EC_value_per_item = (df_twd_pureEC.groupby(['track_user'])['value'].sum() / df_twd_pureEC.groupby(['track_user'])['num_items'].sum()).reset_index()
        # 平均每單金額
        EC_value_per_order = (df_twd_pureEC.groupby(['track_user'])['value'].sum() / df_twd_pureEC.groupby(['track_user'])['num_items'].count()).reset_index()
        # 每月下單頻次
        EC_freq_per_month = (df_twd_pureEC.groupby(['track_user'])['value'].count() / months).reset_index()
        # 下單後距今天數，重複 user_id 則歸納最小值
        df_twd_pureEC['R(days)'] = df_twd_pureEC['pageview_time'].apply(
            lambda x: (datetime.strptime(datetime.now().replace(tzinfo=tw).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')-datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)

        ### 平均每個商品單價
        EC_value_per_item = EC_value_per_item.rename(columns={0: 'value_per_item'}).sort_values('value_per_item')
        dfU_twd_pureEC = df_twd_pureEC.groupby(['track_user'])['ip'].count().reset_index()

        if (v_Q1 == None) or (v_Q2 == None) or (v_Q3 == None):
            v_Q1 = np.quantile(EC_value_per_item['value_per_item'].values, 0.25)
            v_Q2 = np.quantile(EC_value_per_item['value_per_item'].values, 0.5)
            v_Q3 = np.quantile(EC_value_per_item['value_per_item'].values, 0.75)

        c1 = dfU_twd_pureEC['track_user'].isin(EC_value_per_item[EC_value_per_item['value_per_item'] <= v_Q1].track_user.unique())
        c2 = dfU_twd_pureEC['track_user'].isin(EC_value_per_item[(EC_value_per_item['value_per_item'] > v_Q1) & (EC_value_per_item['value_per_item'] <= v_Q2)].track_user.unique())
        c3 = dfU_twd_pureEC['track_user'].isin(EC_value_per_item[(EC_value_per_item['value_per_item'] > v_Q2) & (EC_value_per_item['value_per_item'] <= v_Q3)].track_user.unique())
        c4 = dfU_twd_pureEC['track_user'].isin(EC_value_per_item[EC_value_per_item['value_per_item'] > v_Q3].track_user.unique())

        dfU_twd_pureEC['Avg_Item_Value'] = np.select(condlist=[c1,c2,c3,c4], choicelist=[f'0–{int(round(v_Q1,0))}',f'{int(round(v_Q1,0))}–{int(round(v_Q2,0))}',f'{int(round(v_Q2,0))}–{int(round(v_Q3,0))}',f'{int(round(v_Q3,0))}–'], default=-1)
        dfU_twd_pureEC.drop('ip', axis=1, inplace=True)


        ### 平均每單金額
        EC_value_per_order = EC_value_per_order.rename(columns={0: 'value_per_order'}).sort_values('value_per_order')

        if (aov_Q1 == None) or (aov_Q2 == None) or (aov_Q3 == None):
            aov_Q1 = np.quantile(EC_value_per_order['value_per_order'].values, 0.25)
            aov_Q2 = np.quantile(EC_value_per_order['value_per_order'].values, 0.5)
            aov_Q3 = np.quantile(EC_value_per_order['value_per_order'].values, 0.75)

        c1 = dfU_twd_pureEC['track_user'].isin(EC_value_per_order[EC_value_per_order['value_per_order'] <= aov_Q1].track_user.unique())
        c2 = dfU_twd_pureEC['track_user'].isin(EC_value_per_order[(EC_value_per_order['value_per_order'] > aov_Q1) & (EC_value_per_order['value_per_order'] <= aov_Q2)].track_user.unique())
        c3 = dfU_twd_pureEC['track_user'].isin(EC_value_per_order[(EC_value_per_order['value_per_order'] > aov_Q2) & (EC_value_per_order['value_per_order'] <= aov_Q3)].track_user.unique())
        c4 = dfU_twd_pureEC['track_user'].isin(EC_value_per_order[EC_value_per_order['value_per_order'] > aov_Q3].track_user.unique())
        dfU_twd_pureEC['Avg_Order_Value'] = np.select(condlist=[c1,c2,c3,c4], choicelist=[f'0–{int(round(aov_Q1,0))}',f'{int(round(aov_Q1,0))}–{int(round(aov_Q2,0))}',f'{int(round(aov_Q2,0))}–{int(round(aov_Q3,0))}',f'{int(round(aov_Q3,0))}–'], default=-1)
        dfU_twd_pureEC

        ### 每月下單頻次
        EC_freq_per_month=EC_freq_per_month.rename(columns={'value': 'freq_per_month'})

        if (f_Q1 == None) or (f_Q2 == None) or (f_Q3 == None):
            f_Q1 = 1
            f_Q2 = 2
            f_Q3 = 4

        if months == 1:
            c1 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] == f_Q1].track_user.unique())
            c2 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[(EC_freq_per_month['freq_per_month'] > f_Q1) & (EC_freq_per_month['freq_per_month'] < f_Q3)].track_user.unique())
            c3 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] >= f_Q3].track_user.unique())
            dfU_twd_pureEC['Avg_Order_Frequency'] = np.select(condlist=[c1,c2,c3], choicelist=['每月1次','每月2-3次','每月4次以上'], default=-1)
        else:
            c1 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] < f_Q1].track_user.unique())
            c2 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[(EC_freq_per_month['freq_per_month'] >= f_Q1) & (EC_freq_per_month['freq_per_month'] < f_Q2)].track_user.unique())
            c3 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[(EC_freq_per_month['freq_per_month'] >= f_Q2) & (EC_freq_per_month['freq_per_month'] < f_Q3)].track_user.unique())
            c4 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] >= f_Q3].track_user.unique())
            dfU_twd_pureEC['Avg_Order_Frequency'] = np.select(condlist=[c1,c2,c3,c4], choicelist=['每月不到1次','每月1次', '每月2-3次','每月4次以上'], default=-1)
            
    elif group_num == 2:
        v_Q2 == None
        v_Q3 == None
        EC_value_per_item = (df_twd_pureEC.groupby(['track_user'])['value'].sum() / df_twd_pureEC.groupby(['track_user'])['num_items'].sum()).reset_index()
        EC_value_per_order = (df_twd_pureEC.groupby(['track_user'])['value'].sum() / df_twd_pureEC.groupby(['track_user'])['num_items'].count()).reset_index()
        EC_freq_per_month = (df_twd_pureEC.groupby(['track_user'])['value'].count() / months).reset_index()
        df_twd_pureEC['R(days)'] = df_twd_pureEC['pageview_time'].apply(
            lambda x: (datetime.strptime(datetime.now().replace(tzinfo=tw).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')-datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).days)
        
        ### 平均每個商品單價
        EC_value_per_item = EC_value_per_item.rename(columns={0: 'value_per_item'}).sort_values('value_per_item')
        dfU_twd_pureEC = df_twd_pureEC.groupby(['track_user'])['ip'].count().reset_index()

        if (v_Q1 == None) and (v_Q2 == None) and (v_Q3 == None):
            v_Q1 = np.quantile(EC_value_per_item['value_per_item'].values, 0.5)

        c1 = dfU_twd_pureEC['track_user'].isin(EC_value_per_item[EC_value_per_item['value_per_item'] <= v_Q1].track_user.unique())
        c2 = dfU_twd_pureEC['track_user'].isin(EC_value_per_item[EC_value_per_item['value_per_item'] >= v_Q1].track_user.unique())
        
        dfU_twd_pureEC['Avg_Item_Value'] = np.select(condlist=[c1,c2], choicelist=[f'0–{int(round(v_Q1,0))}',f'{int(round(v_Q1,0))}–'], default=-1)
        dfU_twd_pureEC.drop('ip', axis=1, inplace=True)


        ### 平均每單金額
        EC_value_per_order = EC_value_per_order.rename(columns={0: 'value_per_order'}).sort_values('value_per_order')

        if (aov_Q1 == None) and (aov_Q2 == None) and (aov_Q3 == None):
            aov_Q1 = np.quantile(EC_value_per_order['value_per_order'].values, 0.5)
            
        c1 = dfU_twd_pureEC['track_user'].isin(EC_value_per_order[EC_value_per_order['value_per_order'] <= aov_Q1].track_user.unique())
        c2 = dfU_twd_pureEC['track_user'].isin(EC_value_per_order[EC_value_per_order['value_per_order'] >= aov_Q1].track_user.unique())
        dfU_twd_pureEC['Avg_Order_Value'] = np.select(condlist=[c1,c2], choicelist=[f'0–{int(round(aov_Q1,0))}',f'{int(round(aov_Q1,0))}–'], default=-1)
        dfU_twd_pureEC

        ### 每月下單頻次
        EC_freq_per_month=EC_freq_per_month.rename(columns={'value': 'freq_per_month'})

        if (f_Q1 == None) and (f_Q2 == None) and (f_Q3 == None):
            f_Q1 = 1

        if months == 1:
            c1 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] == f_Q1].track_user.unique())
            c2 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] > f_Q1].track_user.unique())
            dfU_twd_pureEC['Avg_Order_Frequency'] = np.select(condlist=[c1,c2], choicelist=['每月1次','每月2次以上'], default=-1)
        else:
            c1 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] < f_Q1].track_user.unique())
            c2 = dfU_twd_pureEC['track_user'].isin(EC_freq_per_month[EC_freq_per_month['freq_per_month'] >= f_Q1].track_user.unique())
            dfU_twd_pureEC['Avg_Order_Frequency'] = np.select(condlist=[c1,c2], choicelist=['每月不到1次','每月1次以上'], default=-1)


    dfU_twd_pureEC = pd.merge(dfU_twd_pureEC, df_twd_pureEC[['track_user', 'R(days)']].groupby(['track_user']).agg('min').reset_index(), on='track_user', how='left')
#     dfU_twd_pureEC.to_csv(f'pureEC_buyer_labels_{shiftType}_{startDate}_{endDate}.csv')

    detail_size = dfU_twd_pureEC.groupby(['Avg_Item_Value', 'Avg_Order_Value', 'Avg_Order_Frequency']).size().reset_index().rename(columns={0: 'size'}).sort_values('size', ascending=False)#.to_csv('pureEC_buyer_labels_size.csv')
    avgItemV_size = dfU_twd_pureEC.groupby(['Avg_Item_Value']).size().reset_index().rename(columns={0: 'size'})
    avgOrderV_size = dfU_twd_pureEC.groupby(['Avg_Order_Value']).size().reset_index().rename(columns={0: 'size'})
    avgOrderF_size = dfU_twd_pureEC.groupby(['Avg_Order_Frequency']).size().reset_index().rename(columns={0: 'size'})

    return {
            'user_label': dfU_twd_pureEC,
            'detail_size': detail_size, 
            'avgItemV_size': avgItemV_size, 
            'avgOrderV_size': avgOrderV_size,
            'avgOrderF_size': avgOrderF_size
            }


def generate_shifts(df_twd_pureEC, predict_month, for_predict=False):
    if for_predict == False:
        padding = 2
    else:
        padding = 1
    rounds_list = list()
    shift_start_timing = list()
    shift_end_timing = list()
    shift_start = datetime.strptime(f'2020-{predict_month-padding}-{monthrange(2020,predict_month-padding)[1]}', "%Y-%m-%d")
    rounds_list.append(df_twd_pureEC[(df_twd_pureEC['datetime']<=shift_start) & (df_twd_pureEC['datetime']>=shift_start-dt.timedelta(days=30))])
    shift_start_timing.append(shift_start)
    shift_end_timing.append(shift_start-dt.timedelta(days=30))

    for i in tqdm(range(1,10)):
        rounds_list.append(df_twd_pureEC[(df_twd_pureEC['datetime']<=shift_start-dt.timedelta(days=i*14)) & (df_twd_pureEC['datetime']>=shift_start-dt.timedelta(days=i*14+30))])
        shift_start_timing.append(shift_start-dt.timedelta(days=i*14))
        shift_end_timing.append(shift_start-dt.timedelta(days=i*14+30))
        
    return (rounds_list, shift_start, shift_start_timing, shift_end_timing)


def generate_buyer_info_dummy(df_twd_pureEC, shift_start, shift_start_timing, shift_end_timing):
    
    ## 加入 user 有購買 EC/Ind 的 shifts 數
    ## 先跑全部區間的紀錄確認會出現 ec/ind 欄位有哪些
    buyer_info = df_twd_pureEC[['track_user','datetime','industry_id','advertiser_id']]
    buyer_info['industry_id'] = buyer_info['industry_id'].astype(str)
    buyer_info['advertiser_id'] = buyer_info['advertiser_id'].astype(str)
    buyer_info = buyer_info[(buyer_info['datetime']<=shift_start) & (buyer_info['datetime']>=shift_start-dt.timedelta(days=9*14+30))]
    buyer_info_dummy = buyer_info.drop(['datetime'],axis=1)
    buyer_info_dummy = pd.get_dummies(buyer_info_dummy.iloc[:,1:])
    buyer_info_dummy = pd.concat([buyer_info.iloc[:,0], buyer_info_dummy], axis=1)
    buyer_info_dummy = buyer_info_dummy.groupby(['track_user']).sum().reset_index()

    ## 再針對每 shift 有無該 buyer 為參照標記是否當 shift 在 ec/ind 購買
    for c in buyer_info_dummy.columns[1:]:
        buyer_info_dummy[c] = 0.0

    cond_list = [
        (buyer_info['datetime']<=shift_start_timing[0]) & (buyer_info['datetime']>=shift_end_timing[0]),
        (buyer_info['datetime']<=shift_start_timing[1]) & (buyer_info['datetime']>=shift_end_timing[1]),
        (buyer_info['datetime']<=shift_start_timing[2]) & (buyer_info['datetime']>=shift_end_timing[2]),
        (buyer_info['datetime']<=shift_start_timing[3]) & (buyer_info['datetime']>=shift_end_timing[3]),
        (buyer_info['datetime']<=shift_start_timing[4]) & (buyer_info['datetime']>=shift_end_timing[4]),
        (buyer_info['datetime']<=shift_start_timing[5]) & (buyer_info['datetime']>=shift_end_timing[5]),
        (buyer_info['datetime']<=shift_start_timing[6]) & (buyer_info['datetime']>=shift_end_timing[6]),
        (buyer_info['datetime']<=shift_start_timing[7]) & (buyer_info['datetime']>=shift_end_timing[7]),
        (buyer_info['datetime']<=shift_start_timing[8]) & (buyer_info['datetime']>=shift_end_timing[8]),
        (buyer_info['datetime']<=shift_start_timing[9]) & (buyer_info['datetime']>=shift_end_timing[9])
    ]
    choice_list = list(range(10))
    buyer_info['shift'] = np.select(condlist=cond_list, choicelist=choice_list, default=-1)
    buyer_info = buyer_info[buyer_info['shift']!=-1]

    t1 = buyer_info.groupby(['track_user','shift','industry_id'])['datetime'].count().reset_index()
    t2 = buyer_info.groupby(['track_user','shift','advertiser_id'])['datetime'].count().reset_index()

    for ind in tqdm(t1['industry_id'].unique()):
        t = t1[t1['industry_id']==ind]
        for shift in tqdm(range(0, 10)):
            buyer_info_dummy[f'industry_id_{ind}'] = np.where(buyer_info_dummy['track_user'].isin(t[t['shift']==shift]['track_user'].unique()), buyer_info_dummy[f'industry_id_{ind}']+1, buyer_info_dummy[f'industry_id_{ind}'])

    for ec in tqdm(t2['advertiser_id'].unique()):
        t = t2[t2['advertiser_id']==ec]
        for shift in tqdm(range(0, 10)):
            buyer_info_dummy[f'advertiser_id_{ec}'] = np.where(buyer_info_dummy['track_user'].isin(t[t['shift']==shift]['track_user'].unique()), buyer_info_dummy[f'advertiser_id_{ec}']+1, buyer_info_dummy[f'advertiser_id_{ec}'])
            
    return buyer_info_dummy


def unique_user_by_shift(ulabel_MBS2WO_shifts, i, label_cols, label_item):
    userlist_tmp_0 = ulabel_MBS2WO_shifts[0][ulabel_MBS2WO_shifts[0][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_1 = ulabel_MBS2WO_shifts[1][ulabel_MBS2WO_shifts[1][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_2 = ulabel_MBS2WO_shifts[2][ulabel_MBS2WO_shifts[2][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_3 = ulabel_MBS2WO_shifts[3][ulabel_MBS2WO_shifts[3][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_4 = ulabel_MBS2WO_shifts[4][ulabel_MBS2WO_shifts[4][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_5 = ulabel_MBS2WO_shifts[5][ulabel_MBS2WO_shifts[5][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_6 = ulabel_MBS2WO_shifts[6][ulabel_MBS2WO_shifts[6][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_7 = ulabel_MBS2WO_shifts[7][ulabel_MBS2WO_shifts[7][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_8 = ulabel_MBS2WO_shifts[8][ulabel_MBS2WO_shifts[8][label_cols[i]] == label_item]['track_user'].unique()
    userlist_tmp_9 = ulabel_MBS2WO_shifts[9][ulabel_MBS2WO_shifts[9][label_cols[i]] == label_item]['track_user'].unique()

    userlists = [userlist_tmp_0,userlist_tmp_1,userlist_tmp_2,userlist_tmp_3,userlist_tmp_4,userlist_tmp_5,userlist_tmp_6,userlist_tmp_7,userlist_tmp_8,userlist_tmp_9]
    specific_label_all_user = list(set().union(userlist_tmp_0,userlist_tmp_1,userlist_tmp_2,userlist_tmp_3,userlist_tmp_4,userlist_tmp_5,userlist_tmp_6,userlist_tmp_7,userlist_tmp_8,userlist_tmp_9))
    
    return (userlists, specific_label_all_user)




