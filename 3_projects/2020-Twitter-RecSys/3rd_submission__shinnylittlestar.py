import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                "engaging_user_is_verified",
                "engaging_user_account_creation", "engagee_follows_engager", "reply_timestamp", "retweet_timestamp", 
                "retweet_with_comment_timestamp", "like_timestamp"]

values = list()
with open("training.tsv", encoding="utf-8") as f:
    for line in tqdm(f.readlines()[:12434735]):
        line = line.strip()
        data = line.split("\x01")
        values.append(data)

df_train = pd.DataFrame(values, columns=features)
df_train['hashtags'] = df_train['hashtags'].astype(str)  # nan 轉換成 string 進行 Doc2Vec training 時才不會噴錯
print('df_train: ', len(df_train))

df_train['isReply'] = np.where(df_train['reply_timestamp']!='', 1, 0)
df_train['isRetweet'] = np.where(df_train['retweet_timestamp']!='', 1, 0)
df_train['isRetweetComment'] = np.where(df_train['retweet_with_comment_timestamp']!='', 1, 0)
df_train['isLike'] = np.where(df_train['like_timestamp']!='', 1, 0)


val_features = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
                "engaged_with_user_following_count", "engaged_with_user_is_verified",
                "engaged_with_user_account_creation",
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count",
                "engaging_user_is_verified",
                "engaging_user_account_creation", "engagee_follows_engager"]

values = list()
with open("val_v2.tsv", encoding="utf-8") as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        data = line.split("\x01")
        values.append(data)
        
df_val = pd.DataFrame(values, columns=val_features)
df_val['hashtags'] = df_val['hashtags'].astype(str)
print('df_val:', len(df_val))


# 1. `d2v__text_tokens`
documents = [TaggedDocument(doc.split('\t'), [i]) for i, doc in enumerate(df_train['text_tokens'].values)]
model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)
# store to local
fname = get_tmpfile("doc2vec_val")
model.save(fname)
model_texttokens = Doc2Vec.load(fname)

val_text_vector = list()
for i in tqdm(range(0, len(df_val))):
    val_text_vector.append(model_texttokens.infer_vector(df_val['text_tokens'].values[i].split('\t')))
df_val_text_vectors = pd.DataFrame(val_text_vector)
print('df_val_text_vectors:', len(df_val_text_vectors))

train_text_vecs = list()
for i in tqdm(range(0, len(df_train))):
    train_text_vecs.append(model_texttokens.infer_vector(df_train['text_tokens'].values[i].split('\t')))
df_train_text_vectors = pd.DataFrame(train_text_vecs)
print('df_train_text_vectors:', len(df_train_text_vectors))
df_train_text_vectors.columns = ['text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10']
df_val_text_vectors.columns = ['text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10']
df_train = pd.concat([df_train, df_train_text_vectors], axis=1)
df_val = pd.concat([df_val, df_val_text_vectors], axis=1)


# 2. `isHashtag` & `d2v__hashtags`
df_train['isHashtag'] = np.where(df_train['hashtags'] != '', 1, 0)
df_val['isHashtag'] = np.where(df_val['hashtags'] != '', 1, 0)

hashtags_documents = [TaggedDocument(doc.split('\t'), [i]) for i, doc in enumerate(df_train[df_train['isHashtag']==1]['hashtags'].values)]
d2v_model_hashtags = Doc2Vec(hashtags_documents, vector_size=5, window=2, min_count=1, workers=4)
# store to local
fname = get_tmpfile("doc2vec_hashtags")
d2v_model_hashtags.save(fname)
d2v_model_hashtags = Doc2Vec.load(fname)
print('d2v_model_hashtags ready.')

val_hashtag_vector = list()
for i in tqdm(range(0, len(df_val))):
    if df_val['hashtags'].values[i] == 'nan':
        val_hashtag_vector.append([0,0,0,0,0])
    else:
        val_hashtag_vector.append(d2v_model_hashtags.infer_vector(df_val['hashtags'].values[i].split('\t')))
df_val_hashtag_vectors = pd.DataFrame(val_hashtag_vector)

train_hashtag_vecs = list()
for i in tqdm(range(0, len(df_train))):
    if df_train['hashtags'].values[i] == 'nan':
        train_hashtag_vecs.append([0,0,0,0,0])
    else:
        train_hashtag_vecs.append(d2v_model_hashtags.infer_vector(df_train['hashtags'].values[i].split('\t')))
df_train_hashtag_vectors = pd.DataFrame(train_hashtag_vecs)

df_train_hashtag_vectors.columns = ['hashtag_1','hashtag_2','hashtag_3','hashtag_4','hashtag_5']
df_val_hashtag_vectors.columns = ['hashtag_1','hashtag_2','hashtag_3','hashtag_4','hashtag_5']
df_train = pd.concat([df_train, df_train_hashtag_vectors], axis=1)
df_val = pd.concat([df_val, df_val_hashtag_vectors], axis=1)
print('1. isHashtag, 2. d2v__hashtags finished.')


# 3. `vec__present_media`
list__present_media = list()
for item in tqdm(df_train['present_media'].values):
    list_tmp_1 = [item.split('\t').count('Photo'), item.split('\t').count('GIF'), item.split('\t').count('Video')]
    list__present_media.append(list_tmp_1)

df_train_vec__present_media = pd.DataFrame(list__present_media)
df_train_vec__present_media.head()
df_train_vec__present_media.columns = ['media_photo','media_GIF','media_video']

list__present_media = list()
for item in tqdm(df_val['present_media'].values):
    list_tmp_1 = [item.split('\t').count('Photo'), item.split('\t').count('GIF'), item.split('\t').count('Video')]
    list__present_media.append(list_tmp_1)

df_val_vec__present_media = pd.DataFrame(list__present_media)
df_val_vec__present_media.head()
df_val_vec__present_media.columns = ['media_photo','media_GIF','media_video']

df_train_highDen = pd.concat([df_train_text_vectors,df_train_hashtag_vectors],axis=1)
df_train_highDen = pd.concat([df_train_highDen,df_train_vec__present_media],axis=1)
df_val_highDen = pd.concat([df_val_text_vectors,df_val_hashtag_vectors],axis=1)
df_val_highDen = pd.concat([df_val_highDen,df_val_vec__present_media],axis=1)
df_train_highDen.to_csv('training_highDimensionFeatures.csv')
df_val_highDen.to_csv('val_highDimensionFeatures.csv')
del df_train_highDen
del df_val_highDen

df_train = pd.concat([df_train, df_train_vec__present_media], axis=1)
df_val = pd.concat([df_val, df_val_vec__present_media], axis=1)
print('3. vec__present_media finished.')


# # 4. isPresent_links & weighted__domains
# df_train['isPresent_links'] = np.where(df_train['present_links']!='', 1, 0)
# df_val['isPresent_links'] = np.where(df_val['present_links']!='', 1, 0)
#
# df_tmp_3 = df_train.groupby(['present_domains']).size().reset_index()
# df_tmp_3 = df_tmp_3[df_tmp_3['present_domains']!='']
# df_tmp_3.rename(columns={0: 'weighted__domains'}, inplace=True)
# df_train = pd.merge(df_train, df_tmp_3, on='present_domains', how='left')
# df_train = df_train.fillna(0)
# df_train['weighted__domains'] = df_train['weighted__domains'].apply(lambda x: math.log(x+1, 2))  # x+1 to prevent from log function error
#
# df_tmp_3 = df_val.groupby(['present_domains']).size().reset_index()
# df_tmp_3 = df_tmp_3[df_tmp_3['present_domains']!='']
# df_tmp_3.rename(columns={0: 'weighted__domains'}, inplace=True)
# df_val = pd.merge(df_val, df_tmp_3, on='present_domains', how='left')
# df_val = df_val.fillna(0)
# df_val['weighted__domains'] = df_val['weighted__domains'].apply(lambda x: math.log(x+1, 2))  # x+1 to prevent from log function error
# print('4. isPresent_links & weighted__domains finished.')
#
#
# # 5. tweet_type
# df_tmp_4 = pd.get_dummies(df_train[['tweet_type']])
# df_tmp_4.drop('tweet_type_Quote', axis=1, inplace=True)
# df_train = pd.concat([df_train, df_tmp_4], axis=1)
#
# df_tmp_4 = pd.get_dummies(df_val[['tweet_type']])
# df_tmp_4.drop('tweet_type_Quote', axis=1, inplace=True)
# df_val = pd.concat([df_val, df_tmp_4], axis=1)
# print('5. tweet_type finished.')
#
#
# # 6. difference__
# selected_columns = ['engaged_with_user_follower_count','engaged_with_user_following_count','engaging_user_follower_count','engaging_user_following_count']
# for c in selected_columns:
#     df_train[c] = df_train[c].astype(int)
#     df_val[c] = df_val[c].astype(int)
#
# df_train['difference__engaged_user_follow_count'] = df_train['engaged_with_user_follower_count'] - df_train['engaged_with_user_following_count']
# df_train['difference__engaging_user_follow_count'] = df_train['engaging_user_follower_count'] - df_train['engaging_user_following_count']
# df_train['difference__followers_count'] = df_train['engaged_with_user_follower_count'] - df_train['engaging_user_follower_count']
# df_train['difference__following_count'] = df_train['engaged_with_user_following_count'] - df_train['engaging_user_following_count']
# df_train['engaged_with_user_is_verified'] = np.where(df_train['engaged_with_user_is_verified']=='true', 1, 0)
# df_train['engaging_user_is_verified'] = np.where(df_train['engaging_user_is_verified']=='true', 1, 0)
# df_train['engagee_follows_engager'] = np.where(df_train['engagee_follows_engager']=='true', 1, 0)
#
# df_val['difference__engaged_user_follow_count'] = df_val['engaged_with_user_follower_count'] - df_val['engaged_with_user_following_count']
# df_val['difference__engaging_user_follow_count'] = df_val['engaging_user_follower_count'] - df_val['engaging_user_following_count']
# df_val['difference__followers_count'] = df_val['engaged_with_user_follower_count'] - df_val['engaging_user_follower_count']
# df_val['difference__following_count'] = df_val['engaged_with_user_following_count'] - df_val['engaging_user_following_count']
# df_val['engaged_with_user_is_verified'] = np.where(df_val['engaged_with_user_is_verified']=='true', 1, 0)
# df_val['engaging_user_is_verified'] = np.where(df_val['engaging_user_is_verified']=='true', 1, 0)
# df_val['engagee_follows_engager'] = np.where(df_val['engagee_follows_engager']=='true', 1, 0)


# # training preparation
# df_train.drop(['text_tokens','hashtags','present_media','present_links','present_domains', 'tweet_type', 'language',
#              'tweet_timestamp','engaged_with_user_id','engaged_with_user_account_creation','engaging_user_account_creation',
#              'reply_timestamp', 'retweet_timestamp','retweet_with_comment_timestamp', 'like_timestamp'],axis=1,inplace=True)
#
# df_val.drop(['text_tokens','hashtags','present_media','present_links','present_domains', 'tweet_type', 'language',
#              'tweet_timestamp','engaged_with_user_id','engaged_with_user_account_creation','engaging_user_account_creation'],axis=1,inplace=True)
#
# train = df_train.iloc[:,np.r_[1:4,5:9,13:len(df_train.columns)]].to_numpy()
train = df_train[['tweet_id','engaging_user_id','text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','hashtag_1','hashtag_2','hashtag_3','hashtag_4','hashtag_5','media_photo','media_GIF','media_video']]
print('train shape: ', train.shape)
# val = df_val.iloc[:,np.r_[1:4,5:len(df_val.columns)]].to_numpy()
val = df_val[['tweet_id','engaging_user_id','text_1','text_2','text_3','text_4','text_5','text_6','text_7','text_8','text_9','text_10','hashtag_1','hashtag_2','hashtag_3','hashtag_4','hashtag_5','media_photo','media_GIF','media_video']]
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
    df_info.to_csv(f'Prediction_200611_{which}.csv', header=False, index=False)
    print(f'{which} - file saved.')

