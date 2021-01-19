# [Facebook Ads] Ads Stats Prediction
FB Ads optimization managers are struggling make good results for their clients. Can we predict whether the ads performance will drop or boost earlier for better ads adjusting suggestions?
Company: [Analytic Lens](https://jiunyiyang.me/)
Job Title: Founder / Data Scientist


## Performance

- <impressions> R-square: ~0.6

![image](https://i.imgur.com/vn0dLAn.png)


## Dataset Columns
- Date
- Ad_name
- Impressions: times of exposure to the audiences.
- Click_all: times of clicks on post, included link clicks and post clicks.
- Link_clicks: only count in link clicks.
- Website_content_views: after audiences clicked the target link and entered website, times of product detial pageviews.
- Website_searches: times of searches on-site.
- Website_leads: times of login on-site.
- Website_adds_to_cart: times of user add product to carts.
- Website_purchases: times of product purchases.
- Website_purchases_conversion_value: total values of product purchases.
- Cost: Ad spents.



## code walk-through


### Feature Generation
`initial-fb-stats.py`
1. 載入資料
2. 新特徵生成
    - **廣告操作成果指標**
        - `CPA`：`Cost` / `Website_purchases`
        - `ROAS`：`Website_purchases_conversion_value` / `Cost`
        - `CPC`：`Cost` / `Link_clicks`
        - `CPM`：`Cost` / `Impressions` * 1000
            - 被除數若為 0，會得出 np.inf 值；改成補進一個很大的值 (NT$999)
    - **交易時間拆分**
        - `month`：月份
        - `dayOfMonth`：日期
        - `hourOfDay`：當天小時
        - `dayOfWeek`：星期幾
        - `isWeekend`：是否為週末 (bool)
3. 特徵轉換
    - 類別型進行 One-hot 處理
    - 數值欄位進行縮放處理
        - 確保 validation/test set都是用和 train set 一樣的縮放比例


### Make Sequence & Split Dataset
以前一周的各項數據，預測今日的廣告數據；共可生成 總天數-7 的序列組數。
再將資料集以 (train:validation:test = 5:2:3) 比例分割


### Mode Building & Measurement
- LSTM model structure
- Hyperparameters
    - loss: `mean_squared_error`
    - optimizer: `RMSprop`
- Callback: EarlyStopping
- Validation_data: True
- Evaluation: R-square score
    - before evaluate, invert prediction from previous scaler


