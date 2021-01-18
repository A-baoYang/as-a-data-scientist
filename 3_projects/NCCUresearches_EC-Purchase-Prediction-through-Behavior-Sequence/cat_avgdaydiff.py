import pandas as pd
import numpy as np
from tqdm import tqdm


def cat_avgdaydiff(df, cat):

    tmp_df_1 = df[(df['cat_id'] == cat) & (df['purchase'] > 0)].groupby(['user_id'])['day_stamp'].nunique().reset_index().sort_values(['day_stamp'], ascending=False)
    tmp_df_1 = tmp_df_1[tmp_df_1['day_stamp'] > 1]
    cat_repeatbuyer = tmp_df_1.user_id.unique()

    print(f'cat_id: {cat}')
    avgdaydiffs = list()
    for user in tqdm(cat_repeatbuyer):
        tmp_df_2 = df[(df['user_id']==user) & (df['cat_id']==cat)].groupby(['day_stamp'])['purchase'].sum().reset_index().sort_values('day_stamp')
        tmp_list_1 = tmp_df_2[tmp_df_2['purchase'] > 0].day_stamp.values
        
        
        if len(tmp_list_1) < 2:
            print(f'days of {cat}-{user} had bought is less than 2.')
            pass
        
        else:
            avgdaydiff = np.average(tmp_list_1[1:] - tmp_list_1[:-1])
            
            print(f'user_id {user}: {avgdaydiff}')    
            avgdaydiffs.append(avgdaydiff)
    
#     print(f'Avg. re-purchase daydiff in cat_id {cat}: {np.average(avgdaydiffs)}')
#     print(f'Coefficient of Variation of re-purchase daydiff in cat_id {cat}: {np.std(avgdaydiffs) / np.average(avgdaydiffs)}')

    return(np.average(avgdaydiffs))
