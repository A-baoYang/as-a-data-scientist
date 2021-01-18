import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


val_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                "engaging_user_is_verified",
                "engaging_user_account_creation", "engagee_follows_engager"]
#12434735
#31622777*2->RAM limit
train_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                "engaging_user_is_verified",
                "engaging_user_account_creation", "engagee_follows_engager", "reply_timestamp", "retweet_timestamp",
                "retweet_with_comment_timestamp", "like_timestamp"]


# df_train
df_train = pd.DataFrame([list([0]*len(train_features))], columns=train_features)
chunksize = int(round(10 ** 7, 0))
count = 0
for chunk in tqdm(pd.read_csv('training.tsv', chunksize=chunksize, sep='\x01', skipinitialspace=True, names=train_features)):
    df_train = pd.concat([df_train,chunk])
    count += 1
    print(count)
    if count > 2:
        break

df_train = df_train.fillna('')
df_train.drop(['text_tokens', 'hashtags', 'language',
               'tweet_timestamp', 'engaged_with_user_id', 'engaged_with_user_account_creation',
               'engaging_user_account_creation'], axis=1, inplace=True)
df_train = df_train.iloc[1:,:]
print(len(df_train))
print(df_train.head())

df_train['isReply'] = np.where(df_train['reply_timestamp'] != '', 1, 0)
df_train['isRetweet'] = np.where(df_train['retweet_timestamp'] != '', 1, 0)
df_train['isRetweetComment'] = np.where(df_train['retweet_with_comment_timestamp'] != '', 1, 0)
df_train['isLike'] = np.where(df_train['like_timestamp'] != '', 1, 0)
df_train.drop(["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"], axis=1,
              inplace=True)


# df_val

df_val = pd.DataFrame([list([0]*len(val_features))], columns=val_features)
for chunk in tqdm(pd.read_csv('val_v2.tsv', chunksize=chunksize, sep='\x01', skipinitialspace=True, names=val_features)):
    df_val = pd.concat([df_val,chunk])

df_val = df_val.fillna('')
df_val.drop(['text_tokens','hashtags','language',
             'tweet_timestamp','engaged_with_user_id','engaged_with_user_account_creation','engaging_user_account_creation'],axis=1,inplace=True)
df_val = df_val.iloc[1:,:]
print(len(df_val))
print(df_val.head())


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
print('6. difference__ finished.')


# training preparation
df_train.drop(['present_media','present_links','present_domains', 'tweet_type'],axis=1,inplace=True)
df_val.drop(['present_media','present_links','present_domains', 'tweet_type'],axis=1,inplace=True)

df_train.to_csv('training_notextfeatures_5x_val.csv')
df_val.to_csv('val_notextfeatures.csv')

print(df_train.columns)
print(df_val.columns)

# df_train = pd.read_csv('training_notextfeatures_5x_val.csv')
# df_train.drop('Unnamed: 0', axis=1, inplace=True)
# df_val = pd.read_csv('val_notextfeatures.csv')
# df_val.drop('Unnamed: 0', axis=1, inplace=True)
# print(df_train.columns)
# print(df_val.columns)

train = df_train.iloc[:,np.r_[1:4,5:9,13:len(df_train.columns)]].to_numpy()
print('train shape: ', train.shape)
val = df_val.iloc[:,np.r_[1:4,5:len(df_val.columns)]].to_numpy()
print('val shape: ', val.shape)
train_x, test_x, train_y, test_y = train_test_split(train, np.array(df_train[['isReply', 'isRetweet', 'isRetweetComment', 'isLike']].values), random_state=0, test_size=0.33)
print('train_x shape: ', train_x.shape)
print('test_y shape: ', test_y.shape)


# model training & prediction
goals = ['isReply', 'isRetweet', 'isRetweetComment', 'isLike']
for order, which in enumerate(goals):
    train_y_tmp = list()
    test_y_tmp = list()
    for i in tqdm(range(0, len(train_y))):
        train_y_tmp.append(train_y[i][order])
    for i in tqdm(range(0, len(test_y))):
        test_y_tmp.append(test_y[i][order])

    train_y_tmp = np.array(train_y_tmp)
    test_y_tmp = np.array(test_y_tmp)
    class_weight = {0: 1, 1: (len(train_y_tmp) - train_y_tmp.sum()) / train_y_tmp.sum()}
    print(f'{which} class_weight: {class_weight}')

    clf = RandomForestClassifier(max_depth=2, class_weight=class_weight, random_state=0)
    clf.fit(train_x, train_y_tmp)

    # testset performance
    test_x_predict = clf.predict(test_x)
    print(confusion_matrix(test_y_tmp, test_x_predict))
    print(classification_report(test_y_tmp, test_x_predict))

    # predict val.tsv
    val_predict_ = clf.predict(val)
    print('val_predict_: ', len(val_predict_))

    df_info = df_val[['tweet_id', 'engaging_user_id']]
    df_info['Prediction'] = val_predict_
    df_info.to_csv(f'Prediction_4th_200614_{which}.csv', header=False, index=False)
    print(f'{which} - file saved.')

