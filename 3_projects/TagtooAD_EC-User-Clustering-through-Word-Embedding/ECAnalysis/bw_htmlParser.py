import re
import random
import argparse
import logging
import asyncio
import async_timeout
import aiohttp
from datetime import datetime
import dask.dataframe as dd
import numpy as np
import pandas as pd
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import time
from cutWords import *
import os 
import google_storage  #Travis 協助的python 檔案 google_storage.py
import logging


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--semaphore', type=int, required=False, default=5,
                        help='The semaphore counter of asyncio.')
    parser.add_argument('--connection', type=int, required=False, default=25,
                        help='Number of connection for aiohttp client session.')
    parser.add_argument('--date', type=str, required=False, default='20191205',
                        help='Which date product pages are going to be crawled.')
    parser.add_argument('--start', type=int, required=False, default=0,
                        help='Which slice is going to start crawling.')
    args = parser.parse_args()
    return args

def get_meta(meta, html):
    regex = {
        'keywords': '(?<=name="{}" content=")[^"]+', #[^"]
        'title': '(?<=property="og:{}" content=")[^"]+',
        'description': '(?<=property="og:{}" content=")[^"]+',
    }
    regex_meta = regex.get(meta, '').format(meta)
    r = re.search(regex_meta, html)
    if r:
        return r.group()
    else :  
        regex = {
            'keywords': "(?<=var appierRtKeywords = ')[^';]+"}
        regex_meta = regex.get(meta, "").format(meta)
        r = re.search(regex_meta, html)
        return r.group() if r else ""

    

    
    
async def _get(session, i, url, results_list):
    retry = 0
    max_try = 3
    while retry < max_try:
        try:
            with async_timeout.timeout(5):
                headers = {'User-Agent': str(UserAgent().chrome)}
                print(headers)
                async with session.get(url, headers=headers) as resp:
                    assert resp.status == 200 or resp.status == 301, resp.text
                    html = await resp.text() #html
                    asyncio.sleep(15)
                    

                    try:
                        result = (url, get_meta('title', html), get_meta('keywords', html), get_meta('description', html))
                    except Exception as e:
                        print('Error occurred while findimg items: ', e)

                        continue
                    else:
                        results_list.append(result)
                        print('amount crawled: ', len(results_list))
                break
                
        except asyncio.TimeoutError:
            print(f'TimeoutError occurred at {url}. Retry {retry}.')
            retry += 1
            await asyncio.sleep(retry*random.randint(1, 60)) # set random waiting time
        
        except Exception as e:
            print(f'An error occurred at {url}. Message: {e}.')
            break

    return results_list    

    
async def _bound_get(semaphore, session, i, url, results_list):
    async with semaphore:
        return await _get(session, i, url, results_list)
    await semaphore.release()


async def create_connection(semaphore, connection, urls, results_list):
    tasks = []

    # limit the requests
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
    semaphore = asyncio.Semaphore(semaphore)
    connector = aiohttp.TCPConnector(limit=connection, verify_ssl=False)
    
    # keep connection alive for all requests.
    async with aiohttp.ClientSession(connector=connector) as session:
        for i,url in enumerate(urls):
            task = asyncio.ensure_future(_bound_get(semaphore, session, i, url, results_list))
            tasks.append(task)

        await asyncio.gather(*tasks)    

    
def main():
    #決定日期 
    bw_date = '20191101_20191130'
    print(f'Starting HTML parser bw_date ={bw_date}')
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    date = args.date
    start = args.start
       
    s = time.time()    
    df = pd.read_csv(f'gs://tagtoo-bigquery-export/BusinessWeekly/track_content_jieba/bw_track_{bw_date}')
    print(f'資料量: %d'%len(df))
    
    #抓出所有unique文章
    df_t = pd.DataFrame()
    df_t['url'] = df['url'].drop_duplicates().reset_index(drop=True)
    print(f'資料量: %d'%len(df_t))
    
  
   
    #分檔案儲存
    divide = 9
    part = round(len(df_t)/divide)
    for i in range(divide):
        urls = df_t['url'][part*i:part*(i+1)]
    
        results_list = list()
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(create_connection(args.semaphore, args.connection, urls, results_list))
        loop.run_until_complete(future)

        print('url files finished crawling. Storing to files...')
        df = pd.DataFrame(results_list)
        df.columns = ['url','title','keywords','description']
        file_name = f'bw_content_{bw_date}_part_{i}.csv'
        df.to_csv(file_name,index=False)

    print('HTML parser complete.')
    
    #合併檔案儲存    
    df = pd.DataFrame()
    t = dd.read_csv(f'bw_content_{bw_date}_part_*',quoting=3,error_bad_lines=False)
    df = t.compute()
    df.to_csv(f'bw_content_{bw_date}',index=False)


    
    
    
    #開始結巴分詞
    print('Starting words cutting...')
    loop_list = ['title' , 'keywords' , 'description']
    df_jieba = pd.DataFrame(df['url'],columns={'url'})
    for col in loop_list:
        jieba_columns = col
        text_list = jieba_columns_tolist(df,jieba_columns)
        text_jieba_list = jieba_cut(text_list)

        df_jieba[col] = text_jieba_list

    df_jieba.to_csv(f'df_jieba_bw_{bw_date}',index=False)
    print('All Job complete.')
    
    #上傳檔案到GS 
    print('Starting uploading files to Google Storage...')
    GSIO = google_storage.GoogleStorageIO()
    
    file_names = [f'bw_content_{bw_date}',f'df_jieba_bw_{bw_date}']
    for file_name in file_names:
        local_path = os.getcwd()+'/'+ file_name
        path = 'gs://tagtoo-bigquery-export/BusinessWeekly/track_content_jieba/' + file_name
        blob = GSIO.upload_file(gsuri=path, localpath=local_path)
    blob.make_public()
    
    #刪除暫存檔案 
    print('Starting deleting temp files...')
    for i in range(divide):
        file_name = f'bw_content_{bw_date}_part_{i}.csv'
        if os.path.exists(file_name):
            os.remove(file_name)
        else:
            print("The file does not exist")
    print('Complet deleting temp files.')
    
    
           
    
if __name__ == '__main__':
    main()
    
    
#python bw_htmlParser.py