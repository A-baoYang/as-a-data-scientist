import numpy as np 
import pandas as pd
import csv


def industry_dict():
    return {
        0:"其他",
        1:"資訊科技",
        2:"電子3C",
        3:"運動健身",
        4:"服飾配件",
        5:"親子用品",
        6:"進修教育",
        7:"藝文活動",
        8:"養生美容",
        9:"度假旅遊",
        10:"休閒娛樂",
        11:"工商服務",
        12:"政治社福",
        13:"財經金融",
        14:"居家生活",
        15:"房產相關",
        16:"汽機車相關",
        17:"遊戲產業",
        18:"網路服務",
        19:"醫療保健",
        20:"時尚精品",
        21:"食品餐飲",
        22:"寵物相關",
        23:"綜合電商"    
    }

# def ec_chineseName():
#     df_ec_match = pd.read_csv('ec_id_matchlist.csv')
#     df_ec_match = df_ec_match.drop('GA-VIEWID', axis=1)
#     a = df_ec_match['ID']
#     b = df_ec_match['名稱']
#     ec_dict = {}
#     for i, j in zip(a, b):
#         ec_dict.update({i:j})
#     return ec_dict
