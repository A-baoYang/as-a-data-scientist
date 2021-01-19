# Cainiao - Logistics Satisfaction Sequence Clustering (Optimal Matching)
Observing different logistics pattern between different satisfaction level.
Academic: Master of Management Information Systems, National ChengChi University
Professor: [Hao-Chun Chuang (Howard)](https://mis2.nccu.edu.tw/zh_tw/Faculty/Faculty_01/%E8%8E%8A-%E7%9A%93%E9%88%9E-68290166)


## Explanation
1. 物流運送過程會有數個常見事件，從前到後排序為（不一定每個階段都會出現）：
> `CONSIGN` >> `GOT` >> `ARRIVAL` >> `DEPARTURE` >> `SENT_SCAN` >> `SIGNED` >> `TRADE_SUCCESS` >> `FAILURE`
> 
> 其中 `ARRIVAL`和`DEPARTURE`為抵達和離開中繼貨運站。`SIGNED`為確定顧客收到包裹，`TRADE_SUCCESS`為顧客在平台確認已送到，`FAILURE`為送貨失誤。


## Summary
總運送天數越長：
- 滿意度評分較高的：有更多物流事件會集中發生在接近整個運送過程的**尾端**（前面的少數事件延遲時間較久）
- 滿意度評分較低的：有更多物流事件會集中發生在接近整個運送過程的**前端**（後面的少數事件延遲時間較久、給用戶的感受反而較差）


## code walk-through


### Make State-Squences
`Cainiao-action-count-v7-8.ipynb`
兩種策略：
1. 根據時間單位給定 state (有可能漏掉時長較短的 state)
2. 不同的總運送天數分開看，根據順序、捕捉每一個 state


### Optimal Matching
`Cianiao-TraMineR.R`
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
![image](https://i.imgur.com/ruLmvxM.png)

