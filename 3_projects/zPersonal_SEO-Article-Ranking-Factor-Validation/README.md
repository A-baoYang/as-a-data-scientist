# [SEO] Article Ranking Factor Validation
Observing which attributes of articles(website page shows in Google search result) have high correlation with SEO ranking.
Company: [Analytic Lens](https://www.cathaybk.com.tw/cathaybk/)
Job Title: Data Scientist


## Summary

> 注：相關度矩陣會為負數是因為本資料集中以排名為標的，而排名的數值越小越好

- title: 標題中是否包含『seo』 與排名的相關性最高(0.38)
- desc: **內文描述中是否包含『seo』 與排名的相關性最高(0.5)**、其次是包含『公司』(0.3)
- link: 網址中是否包含搜尋詞與排名的相關性偏低 (<0.12)


## code walk-through

### Crawler
`SERP-article-crawler.ipynb`
爬取 Google 搜尋結果頁前5頁文章，包含
- 標題
- 關鍵詞
- 內文描述
- 發布日期
- 文章內容
- 文章網址/網域
- 排名次序
- 標題下方是否出現段落小字
- 標題下方是否出現評價資訊
- 標題下方是否出現問答格式


### Text Cleaning / Spliting / Embedding
`extract-important-keywords+tfidf+word2vec+PCA.ipynb`
1. 載入文章資料
2. 目標值：排名名次
3. 對文章內容進行文本清理
    - Text Cleaning
        - 清除標點
    - Text Spliting 
        - Jieba
        - CKIP (中研院分詞系統)

### Extract Feature from Article
1. TF-IDF
2. Word Embedding
    - Word2Vec


### Correlation
![image](https://i.imgur.com/c0MUyHR.png)
