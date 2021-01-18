DECLARE start_date STRING DEFAULT '20191201';
DECLARE end_date STRING DEFAULT '20191231';
DECLARE ec_id STRING DEFAULT '1253';

WITH tr_join AS (
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
  target_visitor AS (
    SELECT DISTINCT user
    FROM `gothic-province-823.tagtooad.logs_*`, UNNEST(items) items
      WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
      AND items.advertiser = ec_id
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
  bot_user AS (
    SELECT 
      DISTINCT user
    FROM `gothic-province-823.tagtooad.logs_*`, UNNEST(items) items
    WHERE
      _TABLE_SUFFIX BETWEEN start_date AND end_date
      AND SPLIT(IFNULL(user_agent,'0'), ':')[SAFE_OFFSET(0)] = 'bot'
      AND user IS NOT NULL )
      
  SELECT DISTINCT tr.user AS track_user, DATETIME(tr.start_time, "Asia/Taipei") AS pageview_time, EXTRACT(DATE FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS date, EXTRACT(MONTH FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_bymonth, EXTRACT(DAY FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_byday, EXTRACT(DAYOFWEEK FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_byweekday_num, EXTRACT(HOUR FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_byhour, tr.page, tr.type, items.advertiser AS advertiser_id, tr.browser, tr.os, tr.device, tr.page_key, target_purchaser.value, target_purchaser.num_items,
    CASE
        WHEN page LIKE '%utm_source=tagtoo%' THEN 'tagtoo'
        WHEN page LIKE '%utm_source=%' AND page NOT LIKE '%tagtoo%' THEN 'otherAD'
        ELSE 'organic'
      END AS referrer,
    CASE
        WHEN tr.user_agent LIKE 'pc:%' THEN 'pc'
        WHEN tr.user_agent LIKE 'tablet:%' THEN 'tablet'
        WHEN tr.user_agent LIKE 'mobile:%' THEN 'mobile'
        WHEN tr.user_agent LIKE '%bot%' THEN 'bot'
        ELSE 'unknown' 
      END AS deviceCategory
  FROM `gothic-province-823.tagtooad.logs_*` tr, UNNEST(items) AS items
    LEFT JOIN target_purchaser
      ON tr.session = target_purchaser.session 
      AND tr.start_time = target_purchaser.purchase_time
    WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
    AND FORMAT_DATE('%m%d', EXTRACT(DATE FROM tr.start_time)) = _TABLE_SUFFIX
    AND tr.type IN ('track', 'transaction')
    AND tr.user_agent NOT LIKE '%bot%'
    AND SAFE_CAST(items.advertiser AS INT64) IS NOT NULL
    AND SAFE_CAST(items.advertiser AS INT64) < 5000 
    AND tr.user IS NOT NULL
    AND tr.user IN (
      SELECT user FROM target_visitor )
    AND tr.user NOT IN (
      SELECT user FROM excluded_user )
    AND tr.user NOT IN (
      SELECT user FROM bot_user )
)

  SELECT tr_join.*, IFNULL(indust.industry_id, 0) AS industry_id 
  FROM tr_join
    JOIN `gothic-province-823.tagtoo_from_cloudsql.ECID_to_IndustryID` indust 
      ON indust.ec_id = CAST(tr_join.advertiser_id AS INT64)
--       LIMIT 100