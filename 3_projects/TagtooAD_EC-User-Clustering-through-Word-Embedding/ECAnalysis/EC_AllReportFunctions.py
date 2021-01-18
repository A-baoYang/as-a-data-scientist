import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.font_manager import FontProperties
import dask.dataframe as dd
from tqdm import tqdm
import datetime, time, csv
from wordcloud import WordCloud
import jieba
import jieba.analyse
import pickle
fontproperties = FontProperties(fname = 'font_ch.ttf', size = 14)
font = 'font_ch.ttf'
tagtoo_color = ['#E76E5C','#B03F32','#7A070B','#4B3E3B','#231815','#000000']


def queryFromBQ(bigquery_goal):
    s = time.time()
    t = dd.read_csv(bigquery_goal)
    df = t.compute()
    print('Amount of data: ', len(df))
    print(f"Loading Time = {time.time()-s}")
    return df


def countPercentage(df):
    df['percent'] = df['count_user'] / df['count_user'].sum() *100
    return df


def externalweightedPercentage(dfe, df_external_weighted):
    dfe = dfe.merge(df_external_weighted, on='industry_id', how='left')
    dfe['weightedPercent'] = dfe['percent'] * dfe['Weight']
    dfe['newPercent'] = dfe['weightedPercent'] / dfe['weightedPercent'].sum() *100
    dfe = dfe.drop('weightedPercent', axis=1)
    dfe = dfe.rename(columns={'percent': 'origin_percent', 'newPercent': 'percent'})
    return dfe


def allValidPur(df_ind, lower_bound=None, upper_bound=None):
    df_ind_rev = df_ind[(~df_ind['value'].isnull()) & (df_ind['type'] == 'transaction')].sort_values('value', ascending=False)
    df_ind_rev = df_ind_rev[['track_user', 'view_byhour', 'type', 'advertiser_id', 'value', 'num_items']]
    df_ind_rev['aov'] = df_ind_rev['value'] / df_ind_rev['num_items']
    df_ind_rev['advertiser_id'] = df_ind_rev['advertiser_id'].astype(int)
    df_ind_rev['value'] = df_ind_rev['value'].astype(float)
    df_ind_rev['num_items'] = df_ind_rev['num_items'].astype(int)

    # 檢查過每間鞋電商價格，篩掉客單>10000者
    if lower_bound != None:
        df_ind_rev = df_ind_rev[df_ind_rev['aov'] > int(lower_bound)]
    if upper_bound != None:
        df_ind_rev = df_ind_rev[df_ind_rev['aov'] < int(upper_bound)]
    
    return df_ind_rev


def pie_indCompare(client_name, cli_, ind_, title, fontproperties):
    plt.figure(figsize=(6, 8))
    labels = [str(client_name), 'Industry average']     # 製作圓餅圖的類別標籤
    separeted = (0.2, 0)                  # 依據類別數量，分別設定要突出的區塊
    size = [cli_, ind_]   # 製作圓餅圖的數值來源

    plt.pie(size,                           # 數值
            autopct = lambda p: '{:.1f}% '.format(p),
            explode = separeted,            # 設定分隔的區塊位置
            pctdistance = 0.7,              # 數字距圓心的距離
            textprops = {"fontsize" : 16},  # 文字大小           
            shadow = True,                  # 設定陰影
            colors=('firebrick','lightgray'))                  
    plt.axis('equal')                       # 使圓餅圖比例相等
    plt.title(f'{client_name} {title} 佔該產業比例', fontproperties=fontproperties)
    plt.show()


def pie_indCompare_sum(df, cli_id, client_name, col, fontproperties, title='總收益'):
    ind_ = df[str(col)].sum()
    cli_ = df[df['advertiser_id'] == int(cli_id)][str(col)].sum()
    client_name = client_name
    title = title
    pie_indCompare(client_name, cli_, ind_, title, fontproperties)
    

def pie_indCompare_count(df, cli_id, client_name, col, fontproperties, title='總轉換次數'):
    ind_ = df[str(col)].count()
    cli_ = df[df['advertiser_id'] == int(cli_id)][str(col)].count()
    client_name = client_name
    title = title
    pie_indCompare(client_name, cli_, ind_, title, fontproperties)

    
def autolabel(rects, ax, unit_name, fontproperties):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*h, '%1.2f %s'%(float(h), str(unit_name)), ha='center', va='bottom', size=14, fontproperties=fontproperties)


def bar_indCompare(yvals, zvals, unit_name, fontproperties, client_name, figHeight, figWidth):
    N = 1
    x_range = np.arange(N)
    width = 0.2

    fig = plt.figure(figsize=(figHeight, figWidth))
    ax = fig.add_subplot(111)

    rects1 = ax.bar(x_range, yvals, width, color='firebrick')
    rects2 = ax.bar(x_range+width, zvals, width, color='lightgray')

    ax.set_xticks(x_range + width)
    ax.set_xticklabels(' ')
    ax.set_ylabel(unit_name, fontproperties=fontproperties)
    # ax.legend( (rects1[0], rects2[0]), (str(client_name), '產業平均'), prop=fontproperties, loc="upper left", bbox_to_anchor=(0.6,0.5))

    autolabel(rects1, ax, unit_name, fontproperties)
    autolabel(rects2, ax, unit_name, fontproperties)
    plt.show()


def bar_indCompare_(df, fontproperties, cli_id, client_name, col, unit_name, figHeight=3, figWidth=8):
    df_cli = df[df['advertiser_id'] == int(cli_id)]
    yvals = df[str(col)].count() / df[str(col)].nunique()
    zvals = df_cli[str(col)].count() / df_cli[str(col)].nunique()
    bar_indCompare(yvals, zvals, unit_name, fontproperties, client_name, figHeight, figWidth)

    
def bar_indCompare_aov(df, fontproperties, cli_id, client_name, col, unit_name, figHeight=3, figWidth=8):
    df_cli = df[df['advertiser_id'] == int(cli_id)]
    yvals = df[str(col)].sum() / df[str(col)].count()
    zvals = df_cli[str(col)].sum() / df_cli[str(col)].count()
    bar_indCompare(yvals, zvals, unit_name, fontproperties, client_name, figHeight, figWidth)

    
def bar_indCompare_multiDf(df_pv, df_pur, fontproperties, cli_id, client_name, col, unit_name, figHeight=3, figWidth=8):
    df_cli_pv = df_pv[df_pv['advertiser_id'] == int(cli_id)]
    df_cli_pur = df_pur[df_pur['advertiser_id'] == int(cli_id)]
    yvals = len(df_pur[str(col)]) / len(df_pv[str(col)])*100
    zvals = len(df_cli_pur[str(col)]) / len(df_cli_pv[str(col)])*100
    bar_indCompare(yvals, zvals, unit_name, fontproperties, client_name, figHeight, figWidth)

    
def df24hr_count(df):
    df_24hr = df.groupby(['view_byhour']).count()['track_user']
    df_sum = df_24hr.sum()
    return [df_24hr, df_sum]

def df24hr_sum(df, col):
    df_24hr = df.groupby(['view_byhour']).sum()[str(col)]
    df_sum = df_24hr.sum()
    return [df_24hr, df_sum]


def genPercentDf(df_ind, df_cli, client_name):
    df_byhour = pd.DataFrame({
        str(client_name): df24hr_count(df_cli)[0] / df24hr_count(df_cli)[1],
        'Industry': df24hr_count(df_ind)[0] / df24hr_count(df_ind)[1]
    }, columns=[str(client_name), 'Industry'])
    df_byhour['difference'] = df_byhour[client_name] - df_byhour['Industry']
    df_byhour = df_byhour.reset_index()
    return df_byhour


def chart_indCompareByHour(df_pv_byhour, client_name, figWidth=20, figHeight=12, left_y_label='瀏覽量於24小時之百分比', fontproperties=fontproperties, fontsize=20):
    
    figsize = (figWidth, figHeight)
    
    ax = df_pv_byhour[['view_byhour', str(client_name), 'Industry']].groupby(['view_byhour']).sum().plot(kind='line', linewidth=5, colormap='Set1', figsize=figsize, stacked=False, legend=True)
    
    df_pv_byhour[['view_byhour', 'difference']].groupby(['view_byhour']).sum().plot(kind='bar', colormap='Set1', figsize=figsize, stacked=False, alpha=0.1, secondary_y=True, ax=ax, legend=True)
    
    ax.set_ylabel(str(left_y_label), fontproperties=fontproperties, fontsize=fontsize)
    plt.ylabel(f'{client_name} 與產業平均差值', fontproperties=fontproperties, fontsize=fontsize)
    ax.legend([client_name, 'Industry'], prop=fontproperties, fontsize=50, loc='upper left')
    plt.legend(['Difference (right)'], prop=fontproperties, loc='upper left', fontsize=50, bbox_to_anchor=(0,0.93))
    plt.show()

    
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

# find 50% frequency
def findMedian(df, group_col, S):
    s = 0
    tmp = 0
    for i, p in zip(df[str(group_col)], df['percent']):
        tmp += p
        if tmp > int(S):
            s = i
            break
    return s

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

def splitUserGroupDf(df, group_col):
    df_list = []
    df[str(group_col)] = df[str(group_col)].astype(int)
    for i in np.unique(df[str(group_col)].values):
        df_list.append(df[df[str(group_col)] == int(i)])
    return df_list


def printAOVByGroup(df, group_col, lower_bound, upper_bound):
    df_pur = df[df['type'] == 'transaction']
    df_pur = df_pur[(df_pur['value']/df_pur['num_items'] > int(lower_bound)) & (df_pur['value']/df_pur['num_items'] < int(upper_bound))]
    print(df_pur.groupby([str(group_col)])['value'].sum() / df_pur.groupby([str(group_col)])['track_user'].count())
    return df_pur.sort_values('value') # for check


def genGroupDistribution(df, label_col='userGroup', byColumn='industry_id'):
    return df.groupby([str(byColumn), str(label_col)])['track_user'].nunique().reset_index()


def crossIndDisStackedChart(table, industry_dict, fontproperties, byColumn='industry_id', exclude_ind=None, figWidth=20, figHeight=14):
    legend_name = ['主力組','潛力組','猶豫組','路人組','VIP']
    figsize = (figWidth, figHeight)
    itemNum = len(table)
    
    if exclude_ind == None:
        table = table.copy()
    elif exclude_ind != None:
        if type(exclude_ind) == list:

            for ind in exclude_ind:
                table = table[table[str(byColumn)] != int(ind)]
        else:
            table = table[table[str(byColumn)] != int(exclude_ind)]
    
    ax = table.groupby(['industry_id'])['FVFP', 'NVFP', 'FVNP', 'NVNP'].sum().plot(kind='bar', stacked=True, figsize=figsize, position=1, width=0.4, color=tagtoo_color)
    table.groupby(['industry_id'])['VIP'].sum().plot(kind='bar', figsize=figsize, secondary_y=True, position=0, width=0.4, color='white', ax=ax)
    
    if industry_dict != None:
        ax.set_xticklabels([industry_dict[int(i)] for i in table[str(byColumn)]], 
                           fontsize=16, fontproperties=fontproperties, rotation=60)
    else:
        ax.set_xticklabels([int(i) for i in table[str(byColumn)]], 
                           fontsize=16, fontproperties=fontproperties, rotation=60)
        
    plt.xlim([-1, int(itemNum)])
    ax.set_ylabel('用戶分組之不重複人數', fontproperties=fontproperties, fontsize=16)
    plt.ylabel('VIP之不重複人數', fontproperties=fontproperties, fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(legend_name, prop=fontproperties, loc='upper left')
    plt.legend(['VIP'], prop=fontproperties, loc='upper left', bbox_to_anchor=(0,0.825))
    plt.show()

    
# def crossIndDisRelativeChart(df_t, label_col, figsize):
#     df_dis_percent = df_t.groupby(['industry_id', str(label_col)])['track_user'].count() / df_t.groupby(['industry_id'])['track_user'].count()
#     df_dis_percent.unstack().plot(kind='bar', stacked=True, figsize=figsize)
#     plt.show()


# def crossIndDisChart(df, label_col, exclude_ind=None, specific_group=None, figWidth=20, figHeight=12, relative=False):
#     figsize = (figWidth, figHeight)
#     df_t = df.copy()
#     if relative == True:
#         if specific_group != None:
#             print('No percentage stacked bar when specific group was assigned.')
            
#         elif exclude_ind != None:
#             df_t = df_t[df_t['industry_id'] != int(exclude_ind)]
#             crossIndDisRelativeChart(df_t, label_col, figsize)
        
#         else:
#             crossIndDisRelativeChart(df_t, label_col, figsize)
            
#     elif relative == False:
#         if specific_group != None:
#             df_t = df_t[df_t[str(label_col)] == int(specific_group)]
            
#         if exclude_ind != None:
#             df_t = df_t[df_t['industry_id'] != int(exclude_ind)]
    
#         df_t.groupby(['industry_id', str(label_col)])['track_user'].count().unstack().plot(kind='bar', stacked=True, figsize=figsize)

#     else:
#         print('Must assign relative value.')
#     plt.show()

    

    
def genPercent(df, byColumn='industry_id', ec_id=1253):
    df[byColumn] = pd.to_numeric(df[byColumn])
    tr_all = df[df['advertiser_id'] != int(ec_id)].count()['track_user']
    df_percent = pd.DataFrame(df[df['advertiser_id'] != int(ec_id)].groupby(byColumn).count()['track_user']/tr_all *100).reset_index()
    return df_percent

def genCompareDf_userGroup(df_outer_UserGroup, df_outer, byColumn='industry_id'):
    df_compare = genPercent(df_outer_UserGroup[0], str(byColumn)).merge(genPercent(df_outer_UserGroup[1], str(byColumn)), on=str(byColumn), how='outer')
    df_compare = df_compare.merge(genPercent(df_outer_UserGroup[2], str(byColumn)), on=str(byColumn), how='outer')
    df_compare = df_compare.merge(genPercent(df_outer_UserGroup[3], str(byColumn)), on=str(byColumn), how='outer')
    df_compare = df_compare.merge(genPercent(df_outer, str(byColumn)), on=str(byColumn), how='outer')
    df_compare.columns = [str(byColumn), 'FVFP', 'NVFP', 'FVNP', 'NVNP', 'all']
    df_compare = df_compare.fillna(0)
    print(len(df_compare))
    return df_compare

def genCompareDf_vip(df_outer_VIP, df_outer, byColumn='industry_id'):
    df_compare = genPercent(df_outer_VIP[0], str(byColumn)).merge(genPercent(df_outer_VIP[1], str(byColumn)), on=str(byColumn), how='left')
    df_compare = df_compare.merge(genPercent(df_outer, str(byColumn)), on=str(byColumn), how='outer')
    df_compare.columns = [str(byColumn), 'VIP', 'nonVIP', 'all']
    df_compare = df_compare.fillna(0)
    print(len(df_compare))
    return df_compare

def mergeBaseline(df, df_vip, df_baseline, byColumn='advertiser_id'):
    df = df.merge(df_vip, on=str(byColumn), how='outer')
    df = df.merge(df_baseline, on=str(byColumn), how='outer')
    df = df.fillna(0)
    df = df.drop(['all_y', 'count_user'], axis=1)
    df = df.rename(columns={'all_x': 'all', 'percent': 'TagtooDBAvg'})
    return df

def filterMeaningfulItem(df):
    df_S = df[(df['FVFP'] >= 1) | (df['NVFP'] >= 1) | (df['FVNP'] >= 1) | (df['NVNP'] >= 1) | (df['all'] >= 1) | (df['VIP'] >= 1) | (df['nonVIP'] >= 1) | (df['TagtooDBAvg'] >= 1)]
    print(len(df_S))
    return df_S

def addDiffCol(df, baseline_col='TagtooDBAvg', by_col='advertiser_id'):
    for c in df.columns:
        if (c != str(baseline_col)) and (c != str(by_col)):
            df[f'diff_{c}'] = df[str(c)] - df[str(baseline_col)]
            df = df.drop(c, axis=1)
    return df

def absTop20Filter(df, byColumn='advertiser_id', target_col='diff_all'):
    df['abs_diff'] = df['diff_all'].apply(lambda x: abs(x))
    df_F = df.sort_values('abs_diff', ascending=False).head(20)
    df_F = df_F.drop(['abs_diff'], axis=1)
    df_F = df_F.sort_values(str(byColumn))
    return df_F

def pltText(index, df, plt_col):
    for x, y in zip(index, df[plt_col]):
        h = y+0.2 if y>0 else y-0.1  #決定標號的高低
        plt.text(x-0.25,h,'%.1f' %y ,va='top')

def draw_differencePercent(df, byColumn='advertiser_id', ylabel_var='瀏覽', font=fontproperties, industry_dict=None, pltWidth=20, pltHeight=14, bar_width=0.4, itemNum=20):
    print(len(df))
    legend_name = ['主力組','潛力組','猶豫組','路人組','VIP','全站平均']
    figsize = (pltWidth, pltHeight)
    byColumn_ch = '電商編號' if (byColumn == 'advertiser_id') else '產業'
    index = np.arange(itemNum)
    
    fig = plt.figure()
    ax = fig.add_subplot()

    df.groupby([str(byColumn)])['diff_FVFP', 'diff_NVFP', 'diff_FVNP', 'diff_NVNP', 'diff_VIP'].sum().plot(kind='bar', stacked=True, figsize=figsize, ax=ax, position=0, width=bar_width,color = tagtoo_color)
    df.groupby([str(byColumn)])['diff_all'].sum().plot(kind='bar', color='white', figsize=figsize, ax=ax, position=1, width=bar_width, secondary_y=False)

    ax.set_ylabel(f'不重複{ylabel_var}人次 百分比差異', fontsize=20, fontproperties=fontproperties)
    ax.set_xlabel(f'{byColumn_ch}', fontsize=20, fontproperties=fontproperties)
    
    if industry_dict != None:
        plt.xticks(index, [industry_dict[int(i)] for i in df[byColumn]],
                   fontsize=16, fontproperties=fontproperties, rotation=60)
    else:
        plt.xticks(index, [int(i) for i in df[byColumn]],
                   fontsize=16, fontproperties=fontproperties, rotation=60)
        
    plt.xlim([-1, int(itemNum)])
    plt.yticks(fontsize=16)
    plt.legend(legend_name, prop=fontproperties)
    plt.show()

# def chart_intentPercentage_userGroup(num, df_compare, industry_dict=None, byColumn='industry_id', plt1_column='diff_FVFP', plt2_column='diff_NVFP', plt3_column='diff_FVNP', plt4_column='diff_NVNP', compareObjects='全站用戶平均'):
#     n = num
#     if n > 20:
#         df_compare = pd.concat([df_compare.sort_values('diff_NVFP')[:10], df_compare.sort_values('diff_NVFP')[-10:]])
#         n = 20
#     index = np.arange(n)
#     bar_width = 0.2

#     fig = plt.figure(figsize=(20,10))
#     ax = plt.subplot()

#     plt_1 = plt.bar(index, df_compare[plt1_column], bar_width)
#     plt_2 = plt.bar(index+bar_width, df_compare[plt2_column], bar_width)
#     plt_3 = plt.bar(index+bar_width*2, df_compare[plt3_column], bar_width)
#     plt_4 = plt.bar(index+bar_width*3, df_compare[plt4_column], bar_width)

#     plt.legend([plt1_column,plt2_column,plt3_column,plt4_column])
    
#     if industry_dict != None:
#         plt.xticks(index + bar_width*2 , [industry_dict[int(i)] for i in df_compare[byColumn]],
#                    size='large',fontproperties=fontproperties,rotation=60)
#     else:
#         plt.xticks(index + bar_width*2 , [int(i) for i in df_compare[byColumn]],
#                    size='large',fontproperties=fontproperties,rotation=60)
        
#     pltText(index, df_compare, plt1_column)
#     pltText(index, df_compare, plt2_column)
#     pltText(index, df_compare, plt3_column)
#     pltText(index, df_compare, plt4_column)

#     plt.xlabel(byColumn)
#     plt.ylabel('Difference Percentage(%)')
#     plt.title(f'與 {compareObjects} 比較', fontproperties=fontproperties)
#     plt.show()


# def chart_intentPercentage_vip(num, df_compare, industry_dict=None, byColumn='industry_id', plt1_column='diff_VIP', plt2_column='diff_nonVIP', compareObjects='全站用戶平均'):
#     n = num
#     if n > 20:
#         df_compare = pd.concat([df_compare.sort_values('diff_VIP')[:10], df_compare.sort_values('diff_VIP')[-10:]])
#         n = 20
#     index = np.arange(n)
#     bar_width = 0.2

#     fig = plt.figure(figsize=(20,10))
#     ax = plt.subplot()

#     plt_1 = plt.bar(index, df_compare[plt1_column], bar_width)
#     plt_2 = plt.bar(index+bar_width, df_compare[plt2_column], bar_width)

#     plt.legend([plt1_column,plt2_column])
    
#     if industry_dict != None:
#         plt.xticks(index + bar_width*2 , [industry_dict[int(i)] for i in df_compare[byColumn]],
#                    size='large',fontproperties=fontproperties,rotation=60)
#     else:
#         plt.xticks(index + bar_width*2 , [int(i) for i in df_compare[byColumn]],
#                    size='large',fontproperties=fontproperties,rotation=60)
        
#     pltText(index, df_compare, plt1_column)
#     pltText(index, df_compare, plt2_column)

#     plt.xlabel(byColumn)
#     plt.ylabel('Difference Percentage(%)')
#     plt.title(f'與 {compareObjects} 比較', fontproperties=fontproperties)
#     plt.show()

    
def jieba_columns_tolist(df,jieba_columns):
    '''
    將所有title 變成一個list
    支援 'title','keywords','description'

    '''    
    columns = ['title','keywords','description']
    if jieba_columns not in columns : 
        raise ValueError("jieba_columns must be 'title','keywords'or 'description'.")
    
    df[jieba_columns] = df[jieba_columns].fillna("")
    text_list = df[jieba_columns].tolist()
    return text_list


#jieba 處理
def jieba_cut(text_list):
#使用Jieba 分詞
    text_jieba_list = []
    for text in tqdm(text_list):
        text_jieba_list.append(" ".join(jieba.cut(text)))
    return text_jieba_list    
    

#產生文字雲前處理    
def merge_jieba_bwTrack(df,df_t):
    '''
    --INPUT--
    df : 指定時間內的商周的所有Track
    df.columns = ['user', 'url'] 


    df_t : 指定時間內的商周的URL內容(Jieba分詞後)
    df_t.columns = ['url', 'title', 'keywords', 'description']

    --OUTPUT--
    df_m : 指定時間內的商周的使用者看的內容集合(Jieba分詞後)
    df_t.columns = ['user', 'count', 'url_all', 'title_all', 'keywords_all','description_all']
    count : 指定時內間使用者在商周有幾次Track
    '''

    df = pd.merge(df,df_t,on='url',how='left',left_index=True)
    df = df.fillna('')
    
    #將同user 的unique data join 一起 
    df_count = df.groupby('user')['url'].nunique()
    df_title = df.groupby('user')['title'].apply(lambda x : ' / '.join(set(x)))
    df_keywords = df.groupby('user')['keywords'].apply(lambda x : ' / '.join(set(x)))
    df_description = df.groupby('user')['description'].apply(lambda x : ' / '.join(set(x)))
    
    
    
    df_m = pd.DataFrame({'count':df_count,
                      #'url_all':df_url,
                      'title_all': df_title,
                      'keywords_all':df_keywords,
                      'description_all':df_description}
                     ,columns=['count','url_all','title_all','keywords_all','description_all'])
    df_m = df_m.reset_index()
    return df_m
    
    
#產生文字雲
def jieba_generate_wordcloud_freq(df,stopwords_list,font,col='title_all',relative_scaling=0.5,title=""):    
    df = df.reset_index(drop=True)
    all_text = str()
    for i in (range(len(df[col]))):
        all_text = all_text +df[col][i]
    
    
    text = all_text.split()
    #設定不要計算的字
    stopwords = {}.fromkeys(stopwords_list)
    counter = {}
    for i in text:
        if i in stopwords:
            continue
        if i in counter:
            counter[i]+=1
        else:
            counter[i]=1

    font = font
    # With relative_scaling=0, only word-ranks are considered. With relative_scaling=1, a word that is twice as frequent will have twice the size. 
    wc = WordCloud(background_color='white',font_path=font,
                   width = 1440,height = 900 ,relative_scaling=relative_scaling,colormap='gist_heat') 
    wc.generate_from_frequencies(counter)
    
    plt.figure(figsize=(20,20))
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    plt.title(title,fontsize = 20)
    plt.show()
        
   
    df_counter = pd.DataFrame.from_dict(counter, orient='index')
    df_counter.columns = ['count']
    df_counter = df_counter.sort_values('count',ascending = False)
    return counter#df_counter



def draw_group(df,stopwords_list,font,group=3,col='title_all',relative_scaling=1):
    grouplist = [0,1,2,3,'vip']

    if group not in grouplist:
        print("group must be 0,1,2,3 or'vip'")
    elif group == 'vip':
        df = df[df['vip']==1]        
    else:
        df = df[df['userGroup']==group]
    try:
        title = f'User Group = {group}'
        df_counter = jieba_generate_wordcloud_freq(df,stopwords_list,font,col=col,relative_scaling=relative_scaling,title=title)
        return df_counter

    except ValueError:
        print('no word cloud to draw')