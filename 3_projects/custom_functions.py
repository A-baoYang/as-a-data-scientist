import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm


def gen_MonthDayHrMinDayOfWeek(user_clicks, time_col):
    # # 將 日-時-分 拆出來
    user_clicks['month'] = user_clicks[time_col].apply(lambda x: int(x.split(' ')[0].split('-')[1]))
    user_clicks['day'] = user_clicks[time_col].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    user_clicks['hour'] = user_clicks[time_col].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
    user_clicks['minute'] = user_clicks[time_col].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
    user_clicks['day_of_week'] = user_clicks[time_col].apply(lambda x: int(dt.datetime.strptime(x.split(' ')[0], '%Y-%m-%d').weekday()+1))
    user_clicks.drop(time_col, axis=1, inplace=True)
    return user_clicks

def gen_timeblock(df):
    df['timeblock'] = (df['day'] - 1) * 24 + df['hour']
    df = df.sort_values('timeblock')
    return df

