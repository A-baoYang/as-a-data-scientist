DECLARE start_date STRING DEFAULT '20191101';
DECLARE end_date STRING DEFAULT '20191130';

WITH tr_join AS (
  WITH target_purchaser AS (
    SELECT DISTINCT tra.user, tra.value, tra.num_items, tra.purchase_time, tra.session
    FROM `gothic-province-823.tagtoo_transaction.logs_*` tra
      WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
      AND ec_id IN ('256', '285', '861', '868', '916', '1339', '1433', '1463', '1474')
      AND tra.user IS NOT NULL
      AND currency IN ('TWD','NTD')
      AND tra.value > 0
      AND CAST(tra.num_items as float64) > 0
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
      
  SELECT DISTINCT tr.user AS track_user, tr.session, DATETIME(tr.start_time, "Asia/Taipei") AS pageview_time, EXTRACT(HOUR FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_byhour, tr.type, items.advertiser AS advertiser_id, target_purchaser.value, target_purchaser.num_items,
  FROM `gothic-province-823.tagtooad.logs_*` tr, UNNEST(items) AS items
    LEFT JOIN target_purchaser
      ON tr.session = target_purchaser.session 
      AND tr.start_time = target_purchaser.purchase_time
    WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
    AND FORMAT_DATE('%Y%m%d', EXTRACT(DATE FROM tr.start_time)) = _TABLE_SUFFIX
    AND tr.type IN ('track', 'transaction')
    AND items.advertiser IN ('256', '285', '861', '868', '916', '1339', '1433', '1463', '1474')
    AND SAFE_CAST(items.advertiser AS INT64) IS NOT NULL
    AND SAFE_CAST(items.advertiser AS INT64) < 5000 
    AND tr.user IS NOT NULL
    AND tr.user NOT IN (
      SELECT user FROM excluded_user )
    AND tr.user NOT IN (
      SELECT user FROM bot_user )
)

  SELECT tr_join.*, IFNULL(indust.industry_id, 0) AS industry_id 
  FROM tr_join
    JOIN `gothic-province-823.tagtoo_from_cloudsql.ECID_to_IndustryID` indust 
      ON indust.ec_id = CAST(tr_join.advertiser_id AS INT64)
