# Clinical Hypotension Prediction during Hemodialysis 透析中低血壓預測模型研究
合作院方在為病患進行血液透析時，若發生低血壓而未即時處理，可能造成嚴重之血管合併症。
目前困境是，醫護人員為病患量測血壓之頻率最高為 30 分鐘一次。我們能否從血壓量測歷史紀錄、血液透析儀器及穿戴式裝置即時數據中，提前預判低血壓的發生？

Company: [Acusense](http://www.acusense.com.tw/)

Job Title: Data Scientist (part-time)


## Performance

- (label=1) **Precision: 90%**
- (label=1) **Recall: 80%**


## code walk-through

### (I) Data Integration
`preprocessing-format-IV.py`: 自訂最小時間單位，合併血壓歷史數據、血液透析儀器數據及穿戴式裝置數據，統一三份資料的時間單位。
> output: `preprocessing-format-IV.py`

##### Versions
- `format-IV`
- `format-IV.2`: use systole at last measure timing
- `format-IV.3`: use systole at next measure timing, add previous_systole
- `format-IV.4`: 2mins before systole measured, for numeric prediction


### (II) Preprocessing & Feature Generation
`preprocessing-IV&add-delta.ipynb`: 處理合併後產生的缺失值、新增與過去時間點的差值(希望顯示增減趨勢)

1. 缺失值處理
- 三份資料集並非每個時間點都同時涵蓋，此時對缺失時間段，使用**內插法**補值。
- 若是整天都缺資料無法前後對照，則以該位病人其他天的平均值來填補。

2. 新特徵生成
- **前一時間點的血壓值**
    - `previous_`
- **與上一時間點的差值**
    - `delta_`

3. 特徵轉換
- 數值欄位進行縮放處理(標準化)


### (III) Make Sequence / Split & Sampling / Mode Building & Measurement
`LSTM-format-V-Hypo_delta_20.ipynb`: 將上一步合併清理好的資料集，根據自訂的時間窗格，建構出符合 LSTM 模型需求的序列資料；建模、混淆矩陣與收斂圖觀察。

#### Make Sequence / Split & Sampling
1. 使用病人分組策略
為了使模型能泛化應用在不同病患身上，將病患分至 train/validation/test set 互不重複，確保模型不會在兩個不同 split sets 偷學到同一位病患的資料。
> train : validation : test = 3 : 2 : 5

2. 以 data sampling 應對 class imbalance 問題
- Under-sampling (random sample): 將多數類別的資料樣本數降至和少數類別相近
- Over-sampling (SMOTE): 將少數類別的資料樣本數，合成至多數類別相近；使用 KNN 概念，從原始資料點與鄰近資料點取平均，合成標籤為少數類別的新樣本。
- original

#### Mode Building & Measurement
3. LSTM model structure
- Hyperparameters Tuning
    - num of hidden layers: 1, 2
    - num of hidden layer nodes: 1.5 times of input nodes + output nodes
    - num of Dropout layers & ratio: prevent from overfitting
    - Kinds & ratio of regularizers (l1, l2, l1_l2): `l1_l2`
    - Optimizers & Learning Rate: `adam`
    - Loss: `binary_crossentropy`
- Measurement
    - Confusion Metric
    - Loss / Accuracy Graph
    

