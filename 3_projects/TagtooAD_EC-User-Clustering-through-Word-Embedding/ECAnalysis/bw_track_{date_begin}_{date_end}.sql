'''
在{ec_id}的指定時間{date_begin}_{date_end}中
所有使用者瀏覽的url
note : 
已排除 safari只到過一次的使用者
與bot 使用者

儲存檔名 : bw_track_{date_begin}_{date_end}

'''

DECLARE date_begin, date_end , ec_id STRING ;
SET (date_begin, date_end, ec_id) = ('20191001', '20191031', '1347') ;

WITH 
excluded_user AS ( #safari只到過一次的使用者
 SELECT COUNT(*) AS user_count, user
 FROM `gothic-province-823.tagtooad.logs_*`
 WHERE 
    _TABLE_SUFFIX BETWEEN date_begin AND date_end
    AND user_agent LIKE '%safari%'
 GROUP BY user
 HAVING user_count = 1),
 
bot_user AS ( #bot使用者
    SELECT 
      DISTINCT user
    FROM `gothic-province-823.tagtooad.logs_*`, UNNEST(items) items
    WHERE
      _TABLE_SUFFIX BETWEEN date_begin AND date_end
      AND items.advertiser = ec_id 
      AND SPLIT(IFNULL(user_agent,'0'), ':')[SAFE_OFFSET(0)] = 'bot'
      AND user IS NOT NULL )

SELECT
  user,
  REPLACE(REPLACE(SPLIT(ra.page,'&')[SAFE_OFFSET(0)],'ID','id'),'/m/','/') AS url  #把page 變成乾淨的 url

  FROM `gothic-province-823.tagtooad.logs_*` ra, UNNEST(items) AS items
  WHERE 
    _TABLE_SUFFIX BETWEEN date_begin AND date_end #這段時間有到過商周的  
    AND items.advertiser = ec_id #商周
    AND user NOT IN (SELECT user FROM excluded_user)
    AND user NOT IN (SELECT user FROM bot_user)