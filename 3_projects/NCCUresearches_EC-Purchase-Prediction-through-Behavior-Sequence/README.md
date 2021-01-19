# TMall - EC Website Behavior Sequence Clustering (Optimal Matching)
Observing different on-site behavior pattern between different user segmentation (ex: repeat buyers v.s. non-repeat buyers).
Academic: Master of Management Information Systems, National ChengChi University
Professor: [Hao-Chun Chuang (Howard)](https://mis2.nccu.edu.tw/zh_tw/Faculty/Faculty_01/%E8%8E%8A-%E7%9A%93%E9%88%9E-68290166)


## Explanation
1. 電商購物前，用戶會有數個常見瀏覽事件（排序不依定、不一定各用戶都會出現每種事件）：
> `click`(pageview), `add_to_cart`, `add_to_favorite`, `purchase`
> 
> user_log 檔案紀錄每個動作發生的時間戳記

2. 電商商品屬性：品類、商家、品牌
3. 用戶屬性：年齡、性別


## Summary
- 看越多品類的用戶，相對更高機率購買


## code walk-through

### Make State-Squences
`TMall_user_states_REdefine_V3.ipynb`
選取兩種對立的族群比較：
1. 某月份有買 v.s. 沒買
2. 總期間曾回購者(購買2+次) v.s. 總期間只買過一次


### Optimal Matching
`TMall-TraMineR.R`
使用R套件：`TraMineR`
1. 定義 states
2. 計算 states 間的 substitution cost 及 indel cost ＜可能隨著 states 或其在序列中位置而有不同＞
    - substitution cost: 從某一 state 改成另一 state 所需的成本
    - indel cost: 刪除一個 state 或在 state 間插入一個空白所需的成本
3. 計算 sequences 間的 dissimilarity (pairwise，形成一個 序列總數x序列總數 的矩陣)
4. 以 dissimilarity matrix 跑 clustering methods:
    - 聚合式階層分群法（agglomerative hierarchical clustering）- Ward's method
    - k-means clustering
5. 畫出各群序列的 state distribution (chronograms)
example:
![image](https://i.imgur.com/37W76x0.png)

