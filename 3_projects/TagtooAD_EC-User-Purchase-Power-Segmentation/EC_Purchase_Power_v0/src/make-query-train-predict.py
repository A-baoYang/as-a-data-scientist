import os
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")
import upload_GoogleStorage as upload_gs

## google storage auth 
# GSIO = google_storage.GoogleStorageIO()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "application_default_credentials.json"
bq_project = 'gothic-province-823'
bq_bucket = 'tagtoo-bigquery-export'
base_folder ='pureEC_timeShifting_validation'
gcs_base_folder = f'ECAnalysis/{base_folder}'
dataset_id = 'tagtoo_export_results'
table_id = 'pureEC_purchase_power_training_tmp_table'
output = 'gs://tagtoo-bigquery-export/ECAnalysis/pureEC_timeShifting_validation/'

OUTPUTS = {
    'output': 'sql_query.sql'
}

# DECLARE media_active_date STRING DEFAULT '20200101';
# DECLARE media_deactive_date STRING DEFAULT '20200531'; # 取直至5/31的工商讀者
# DECLARE from_date STRING DEFAULT '20190301'; # 瀏覽紀錄區間
# DECLARE to_date STRING DEFAULT '20200301'; # 瀏覽紀錄區間
# DECLARE media_ec STRING DEFAULT '1604'; # 篩去過工商者在工商以外的紀錄
# DECLARE non_pure_ec_id_list ARRAY <string>;
# SET non_pure_ec_id_list = ['100','708','787','1201','1202','1204','1305','1497','1538','1555','155','292','387','824','1213'];

#排除非純電商
non_pure_ec_id_list = ['100','708','787','1201','1202','1204','1305','1497','1538','1555','155','292','387','824','1213'] 

def get_query(from_date, to_date, media_ec, media_active_date, media_deactive_date=None, excluded_ec=non_pure_ec_id_list, query_limit=None):
    
    if media_active_date == None:
        media_active_date = from_date
    if media_deactive_date == None:
        media_deactive_date = to_date
        
    print(f'Query from {from_date} to {to_date}...')
    sql = f"""
WITH tr_join AS (
  WITH target_purchaser AS (
    SELECT DISTINCT tra.user, tra.value, tra.currency, tra.num_items, tra.content_ids, tra.purchase_time, tra.session, tra.ip
    FROM `gothic-province-823.tagtoo_transaction.logs_*` tra
      WHERE _TABLE_SUFFIX BETWEEN '{from_date}' AND '{to_date}'
      AND tra.user IS NOT NULL
--       AND currency IN ('TWD','NTD')
--       AND tra.value > 0
      AND CAST(tra.num_items as float64) > 0
      AND user IS NOT NULL
  ),
  media_user AS (
    SELECT DISTINCT tr.user AS user, advertiser
    FROM `gothic-province-823.tagtooad.logs_*` tr, UNNEST(items) items
      WHERE _TABLE_SUFFIX BETWEEN '{media_active_date}' AND '{media_deactive_date}'
      AND advertiser = '{media_ec}'
--       AND user IS NOT NULL
  ),
  excluded_user AS (
    SELECT COUNT(*) user_count, user
    FROM `gothic-province-823.tagtooad.logs_*`, UNNEST(items) items
      WHERE _TABLE_SUFFIX BETWEEN '{media_active_date}' AND '{media_deactive_date}'
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
      _TABLE_SUFFIX BETWEEN '{media_active_date}' AND '{media_deactive_date}'
      AND SPLIT(IFNULL(user_agent,'0'), ':')[SAFE_OFFSET(0)] = 'bot'
      AND user IS NOT NULL )
      
  SELECT DISTINCT tr.user AS track_user, DATETIME(tr.start_time, "Asia/Taipei") AS pageview_time, EXTRACT(DATE FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS date, EXTRACT(MONTH FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_bymonth, EXTRACT(DAY FROM tr.start_time AT TIME ZONE "Asia/Taipei") AS view_byday, tr.page, tr.type, items.advertiser AS advertiser_id,  tr.page_key, target_purchaser.value, target_purchaser.num_items, target_purchaser.content_ids, target_purchaser.currency, target_purchaser.ip,
    CASE
        WHEN tr.user_agent LIKE 'pc:%' THEN 'pc'
        WHEN tr.user_agent LIKE 'tablet:%' THEN 'tablet'
        WHEN tr.user_agent LIKE 'mobile:%' THEN 'mobile'
        WHEN tr.user_agent LIKE '%bot%' THEN 'bot'
        ELSE 'unknown' 
      END AS deviceCategory
  FROM `gothic-province-823.tagtooad.logs_*` tr, UNNEST(items) AS items
    JOIN target_purchaser
      ON tr.session = target_purchaser.session 
      AND tr.start_time = target_purchaser.purchase_time
    WHERE _TABLE_SUFFIX BETWEEN '{from_date}' AND '{to_date}'
    AND FORMAT_DATE('%Y%m%d', EXTRACT(DATE FROM tr.start_time)) = _TABLE_SUFFIX
    AND tr.user IN (
      SELECT user FROM media_user )
    AND items.advertiser != '{media_ec}'
    AND items.advertiser NOT IN UNNEST({excluded_ec})
    AND tr.type IN ('track', 'transaction')
    AND tr.user_agent NOT LIKE '%bot%'
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
    LEFT JOIN `gothic-province-823.tagtoo_from_cloudsql.ECID_to_IndustryID` indust 
      ON indust.ec_id = CAST(tr_join.advertiser_id AS INT64)
"""
    if query_limit:
        sql = sql + f'LIMIT {query_limit}'
    return sql
        

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_date', type=int, required=True, 
                        help='Start date of training set.')
    parser.add_argument('--to_date', type=int, required=True, 
                        help='End date of training set.')
    parser.add_argument('--media_id', type=int, required=True, 
                        help='Target media for generating TTD label.')
    parser.add_argument('--media_active_date', type=int, required=False, 
                        help='Start date of target media installed Tagtoo tracking, or default will set as same as "from date".')
    parser.add_argument('--media_deactive_date', type=int, required=False, 
                        help='End date of target media installed Tagtoo tracking, or default will set as same as "to date".')
    parser.add_argument('--query_limit', type=int, required=False, 
                        help='Query limit in BigQuery.')
    args = parser.parse_args()
    return args


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    
    if args.from_date and args.to_date:
        query = get_query(args.from_date, args.to_date, args.media_id, args.media_active_date, args.media_deactive_date, non_pure_ec_id_list, args.query_limit)
        output_path = output + f'1604visitor_in_pureEC_{args.from_date}_{args.to_date}/'
        upload_gs.sent_query(query, bq_project, dataset_id, table_id, output_path)
    else:
        raise NotImplementedError('(--from_date, --to_date) is required.')
        
    
    with open(OUTPUTS['output'], 'w') as f:
        f.write(query)
        
    print('Job completed.')

    
if __name__ == '__main__':
    main()
        