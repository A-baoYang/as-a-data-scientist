DECLARE start_date STRING DEFAULT '20191225';
DECLARE end_date STRING DEFAULT '20191231';
DECLARE ec_id STRING DEFAULT '1039';

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
)

SELECT DISTINCT tr.user AS track_user, DATETIME(tr.start_time, "Asia/Taipei") AS pageview_time, tr.type, items.advertiser AS advertiser_id, target_purchaser.value, target_purchaser.num_items
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
    