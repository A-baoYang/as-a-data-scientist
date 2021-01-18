import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                "engaging_user_is_verified",
                "engaging_user_account_creation", "engagee_follows_engager", "reply_timestamp", "retweet_timestamp", 
                "retweet_with_comment_timestamp", "like_timestamp"]
values = list()
with open('training.tsv', encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        data = line.split("\x01")
        values.append(data)

df_train = pd.DataFrame(values, columns=features)
df_train.drop(['text_tokens', 'hashtags', 'language',
               'tweet_timestamp', 'engaged_with_user_id', 'engaged_with_user_account_creation',
               'engaging_user_account_creation'], axis=1, inplace=True)
print('df_train: ', len(df_train))

df_train['isReply'] = np.where(df_train['reply_timestamp'] != '', 1, 0)
df_train['isRetweet'] = np.where(df_train['retweet_timestamp'] != '', 1, 0)
df_train['isRetweetComment'] = np.where(df_train['retweet_with_comment_timestamp'] != '', 1, 0)
df_train['isLike'] = np.where(df_train['like_timestamp'] != '', 1, 0)
df_train.drop(["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"], axis=1,
              inplace=True)


val_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                "engaging_user_is_verified",
                "engaging_user_account_creation", "engagee_follows_engager"]


values = list()
with open('val_v2.tsv', encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        data = line.split("\x01")
        values.append(data)

df_val = pd.DataFrame(values, columns=val_features)
df_val.drop(['text_tokens','hashtags','language',
             'tweet_timestamp','engaged_with_user_id','engaged_with_user_account_creation','engaging_user_account_creation'],axis=1,inplace=True)
print('df_val:', len(df_val))


# 4. isPresent_links & weighted__domains
df_train['isPresent_links'] = np.where(df_train['present_links']!='', 1, 0)
df_val['isPresent_links'] = np.where(df_val['present_links']!='', 1, 0)

df_tmp_3 = df_train.groupby(['present_domains']).size().reset_index()
df_tmp_3 = df_tmp_3[df_tmp_3['present_domains']!='']
df_tmp_3.rename(columns={0: 'weighted__domains'}, inplace=True)
df_train = pd.merge(df_train, df_tmp_3, on='present_domains', how='left')
df_train = df_train.fillna(0)
df_train['weighted__domains'] = df_train['weighted__domains'].apply(lambda x: math.log(x+1, 2))  # x+1 to prevent from log function error

df_tmp_3 = df_val.groupby(['present_domains']).size().reset_index()
df_tmp_3 = df_tmp_3[df_tmp_3['present_domains']!='']
df_tmp_3.rename(columns={0: 'weighted__domains'}, inplace=True)
df_val = pd.merge(df_val, df_tmp_3, on='present_domains', how='left')
df_val = df_val.fillna(0)
df_val['weighted__domains'] = df_val['weighted__domains'].apply(lambda x: math.log(x+1, 2))  # x+1 to prevent from log function error
print('4. isPresent_links & weighted__domains finished.')


# 5. tweet_type
df_tmp_4 = pd.get_dummies(df_train[['tweet_type']])
df_tmp_4.drop('tweet_type_Quote', axis=1, inplace=True)
df_train = pd.concat([df_train, df_tmp_4], axis=1)

df_tmp_4 = pd.get_dummies(df_val[['tweet_type']])
df_tmp_4.drop('tweet_type_Quote', axis=1, inplace=True)
df_val = pd.concat([df_val, df_tmp_4], axis=1)
print('5. tweet_type finished.')


# 6. difference__
selected_columns = ['engaged_with_user_follower_count','engaged_with_user_following_count','engaging_user_follower_count','engaging_user_following_count']
for c in selected_columns:
    df_train[c] = df_train[c].astype(int)
    df_val[c] = df_val[c].astype(int)

df_train['difference__engaged_user_follow_count'] = df_train['engaged_with_user_follower_count'] - df_train['engaged_with_user_following_count']
df_train['difference__engaging_user_follow_count'] = df_train['engaging_user_follower_count'] - df_train['engaging_user_following_count']
df_train['difference__followers_count'] = df_train['engaged_with_user_follower_count'] - df_train['engaging_user_follower_count']
df_train['difference__following_count'] = df_train['engaged_with_user_following_count'] - df_train['engaging_user_following_count']
df_train['engaged_with_user_is_verified'] = np.where(df_train['engaged_with_user_is_verified']=='true', 1, 0)
df_train['engaging_user_is_verified'] = np.where(df_train['engaging_user_is_verified']=='true', 1, 0)
df_train['engagee_follows_engager'] = np.where(df_train['engagee_follows_engager']=='true', 1, 0)

df_val['difference__engaged_user_follow_count'] = df_val['engaged_with_user_follower_count'] - df_val['engaged_with_user_following_count']
df_val['difference__engaging_user_follow_count'] = df_val['engaging_user_follower_count'] - df_val['engaging_user_following_count']
df_val['difference__followers_count'] = df_val['engaged_with_user_follower_count'] - df_val['engaging_user_follower_count']
df_val['difference__following_count'] = df_val['engaged_with_user_following_count'] - df_val['engaging_user_following_count']
df_val['engaged_with_user_is_verified'] = np.where(df_val['engaged_with_user_is_verified']=='true', 1, 0)
df_val['engaging_user_is_verified'] = np.where(df_val['engaging_user_is_verified']=='true', 1, 0)
df_val['engagee_follows_engager'] = np.where(df_val['engagee_follows_engager']=='true', 1, 0)


# training preparation
df_train.drop(['present_media','present_links','present_domains', 'tweet_type'],axis=1,inplace=True)
df_val.drop(['present_media','present_links','present_domains', 'tweet_type'],axis=1,inplace=True)

df_train.to_csv('training_notextfeatures_1x_val.csv')
df_val.to_csv('val_notextfeatures.csv')

print(df_train.columns)
print(df_val.columns)

