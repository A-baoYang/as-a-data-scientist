#試圖在一個query 做完label
#VIP 定義: 在所有sum_vale 購買總金額大於0的前20％使用者

DECLARE start_date STRING DEFAULT '20191001';
DECLARE end_date STRING DEFAULT '20191231';
DECLARE ec_id STRING DEFAULT '1039';
DECLARE vip_treshold INT64 DEFAULT 20 ;

WITH target_purchaser AS (
    SELECT DISTINCT tra.user, tra.value, tra.num_items, tra.purchase_time, tra.session
    FROM `gothic-province-823.tagtoo_transaction.logs_*` tra
        WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
        AND tra.user IS NOT NULL
        AND currency IN ('TWD','NTD')
        AND tra.value > 0
        AND CAST(tra.num_items as float64) > 0
        AND user IS NOT NULL
        AND tra.ec_id = ec_id
),
unlabeled_user AS (
  SELECT 
    DISTINCT tr.user AS user
    ,(COUNT(IF(tr.type = 'track', tr.user,null)) + COUNT(IF(tr.type = 'transaction', tr.user,null))) AS track_count #type = transaction 也算是一個track
    ,COUNT(IF(tr.type = 'transaction' ,tr.user,null)) AS tran_count
    ,SUM(target_purchaser.value) AS sum_value
  FROM `gothic-province-823.tagtooad.logs_*` tr, UNNEST(items) AS items
  LEFT JOIN target_purchaser
      ON tr.session = target_purchaser.session 
      AND tr.start_time = target_purchaser.purchase_time
  WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
      AND items.advertiser = ec_id
      AND tr.type IN ('track', 'transaction')
      AND SAFE_CAST(items.advertiser AS INT64) IS NOT NULL
      AND SAFE_CAST(items.advertiser AS INT64) < 5000
      AND tr.user IS NOT NULL
      AND user_agent NOT LIKE '%safari%'
      AND user_agent NOT LIKE '%bot%'  
  GROUP BY 1),
vars AS(
  SELECT 
    APPROX_QUANTILES(track_count,10)[OFFSET(5)] as track_treshold, #取中位數
    APPROX_QUANTILES(tran_count,10)[OFFSET(5)] as tran_treshold #取中位數
  FROM unlabeled_user
),
transaction_user AS (
  SELECT 
    user, sum_value 
  FROM unlabeled_user
  WHERE sum_value > 0
),
ranked_transaction_user AS (
  SELECT user , (RANK() OVER ( ORDER BY sum_value DESC)) / (SELECT COUNT(*) FROM transaction_user) * 100 AS pct
  FROM transaction_user
),
labeled_user AS (
  SELECT *,
  CASE
   WHEN track_count > (SELECT track_treshold FROM vars) AND tran_count > (SELECT tran_treshold FROM vars) THEN '0'
   WHEN track_count <=  (SELECT track_treshold FROM vars) AND tran_count > (SELECT tran_treshold FROM vars) THEN '1'
   WHEN track_count > (SELECT track_treshold FROM vars) AND tran_count <= (SELECT tran_treshold FROM vars) THEN '2'
   #WHEN track_count <= (SELECT track_treshold FROM vars) AND tran_count <= (SELECT tran_treshold FROM vars) THEN '3'
   ELSE 'other'
  END AS user_group,
  CASE 
    WHEN user IN (SELECT user FROM ranked_transaction_user WHERE pct < vip_treshold) THEN TRUE
    ELSE FALSE
  END AS vip
  FROM unlabeled_user)

SELECT * #user, user_group, vip
FROM labeled_user
WHERE user_group != 'other'



/*
#VIP 定義: 在所有sum_vale 購買累積金額前20％的使用者
           
DECLARE start_date STRING DEFAULT '20191225';
DECLARE end_date STRING DEFAULT '20191231';
DECLARE ec_id STRING DEFAULT '1039';
DECLARE track_treshold INT64 DEFAULT 2 ;
DECLARE tran_treshold INT64 DEFAULT 1 ;
DECLARE vip_treshold INT64 DEFAULT 80 ;

WITH target_purchaser AS (
    SELECT DISTINCT tra.user, tra.value, tra.num_items, tra.purchase_time, tra.session
    FROM `gothic-province-823.tagtoo_transaction.logs_*` tra
        WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
        AND tra.user IS NOT NULL
        AND currency IN ('TWD','NTD')
        AND tra.value > 0
        AND CAST(tra.num_items as float64) > 0
        AND user IS NOT NULL
),
excluded_user AS (
    SELECT COUNT(*) user_count, user
    FROM `gothic-province-823.tagtooad.logs_*`, UNNEST(items) items
        WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
        AND user_agent LIKE '%safari%'
        AND user IS NOT NULL
        GROUP BY user
        HAVING user_count = 1
),
unlabeled_user AS (
  SELECT 
     DISTINCT tr.user AS user
    ,COUNT(IF(tr.type = 'track' ,tr.user,null)) AS track_count
    ,COUNT(IF(tr.type = 'transaction' ,tr.user,null)) AS tran_count
    ,SUM(target_purchaser.value) AS sum_value
    ,SUM(SUM(target_purchaser.value)) OVER ( ORDER BY SUM(target_purchaser.value) ASC rows between unbounded preceding and current row) AS cum_sum    
  FROM `gothic-province-823.tagtooad.logs_*` tr, UNNEST(items) AS items
  LEFT JOIN target_purchaser
      ON tr.session = target_purchaser.session 
      AND tr.start_time = target_purchaser.purchase_time
  WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
      AND items.advertiser = ec_id
      AND tr.type IN ('track', 'transaction')
      AND tr.user_agent NOT LIKE '%bot%'
      AND SAFE_CAST(items.advertiser AS INT64) IS NOT NULL
      AND SAFE_CAST(items.advertiser AS INT64) < 5000 
      AND tr.user IS NOT NULL
      AND tr.user NOT IN (
        SELECT user FROM excluded_user )      
  GROUP BY 1),

cum_user AS(
SELECT *
,(cum_sum / (SELECT SUM(sum_value) FROM unlabeled_user)*100) AS cum_pct
FROM unlabeled_user)

SELECT *,
CASE
 WHEN track_count >= track_treshold AND tran_count >= tran_treshold THEN '0'
 WHEN track_count < track_treshold AND tran_count >= tran_treshold THEN '1'
 WHEN track_count >= track_treshold AND tran_count < tran_treshold THEN '2'
 WHEN track_count < track_treshold AND tran_count < tran_treshold THEN '3'
 ELSE 'other'
END AS user_group,
CASE 
 WHEN cum_pct >= vip_treshold THEN TRUE
 ELSE FALSE
END AS vip
FROM cum_user
ORDER BY tran_count DESC

*/

