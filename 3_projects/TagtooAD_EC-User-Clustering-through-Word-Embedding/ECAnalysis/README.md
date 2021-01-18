# ECAnalysis

`EC_AllReportFunctions.py` - 集中所有ECAnalysis Report使用到的Functions（功能說明待補）

  - [x] queryFromBQ
  - [x] allValidPur
  - [x] pie_indCompare_
  - [x] bar_indCompare_
  - [x] df24hr_
  - [x] genPercentDf
  - [x] chart_indCompareByHour
  - [x] freqTable
  - [x] reverseFreqTable
  - [x] findMedian
  - [x] labelUserGroup
  - [x] printAOVByGroup
  - [x] crossIndDisChart
  - [x] genPercent
  - [x] genCompareDf_
  - [x] pltText
  - [x] chart_intentPercentage_
  

`ECTemplateExample.ipynb` - 產出完整ECAnalysis Report的使用範例

> import `EC_AllReportFunctions` as ec


## 產生商周文字雲

### Input 
1. {bw_date} 決定使用商週哪段時間的瀏覽行為
2. `bw_track_{bw_date}` 來自query `bw_track_{date_begin}_{date_end}.sql`
3. 5群Label 過的指定EC的user id 
像是 : `1253_Visitor_201911_internal_viewRecords.csv`

### output
1. 商周在指定{bw_date}中 的文章 ['url','title' , 'keywords' , 'description']`bw_content_{bw_date}`
2. 商周文章Jieba後的DataFrame`df_jieba_bw_content_{bw_date}`
3. 5群TA [0: 主力組 , 1: 潛力組 , 2: 猶豫組 , 3: 路人組  ,'vip' : VIP組] 的文字雲

### Files
1. Bigquery `bw_track_{date_begin}_{date_end}.sql`
2. htmlParser `bw_htmlParser.py` 
3. reportOutput `EC_AllReportFunctions.py`


### 範例
1. `bw_htmlParser.py` 
需要在 main() 裡面決定 {bw_date} 
需要先在Google storage 有了對應的 `bw_track_{bw_date}` 檔案 (query `bw_track_{date_begin}_{date_end}.sql`)
在 Terminal 跑`python bw_htmlParser.py` 會開始爬取商周在指定{bw_date}中 的文章 ['title' , 'keywords' , 'description']
分9個檔案儲存後再合併一個檔案存成 `bw_content_{bw_date}`
執行Jieba 存成 `df_jieba_bw_content_{bw_date}`
處理完後的檔案放在 `gs://tagtoo-bigquery-export/BusinessWeekly/track_content_jieba/`
2. `testResult_wordcloud.ipynb`  

## Enviornment
1. 安裝檔 `requirement.txt`
2. 在 Terminal 輸入 `conda install --file requirements.txt --yes` 
3. 在 Terminal 輸入 `conda install -c mlgill fake-useragent --yes`  (待補充)



# 產業加權
1. 參考資料 [link](https://docs.google.com/spreadsheets/d/1c0o2KKvVawax4314dgthSYz5tOJRC8_hb7ELt9J_0bc/edit?folder=0AE5aOHdYzhXXUk9PVA#gid=1076512551)
2. path `gs://tagtoo-bigquery-export/ECAnalysis/src/industry_weight.csv`