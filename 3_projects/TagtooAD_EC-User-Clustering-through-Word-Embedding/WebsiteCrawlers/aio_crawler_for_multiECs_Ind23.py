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

def parse_rules(html, url):
    if 'momo' in str(url):
        ec_id = 1039
        try:
            l = re.findall('[\w]+', url)
            prod_id = l[l.index('i_code')+1]
        except:
            prod_id = ''
        try:
            prod_title = html.find(class_='prdnoteArea').h1.text
        except:
            try:
                prod_title = html.find(id='goodsName').text # if mobile page
            except:
                prod_title = ''
        try:
            prod_breadcrumbs = html.find(id='bt_2_layout_NAV').text.replace('\n', ' ')
        except:
            try:
                prod_breadcrumbs = html.find(class_='pathArea').text.replace('\n', ' ') # if mobile page
            except:
                prod_breadcrumbs = ''

    elif 'friday' in str(url):
        ec_id = 1159
        try:
            l = re.findall('[\w]+', url)
            prod_id = l[l.index('pid')+1]
        except:
            prod_id = ''
        try:
            prod_title = html.find(id='flag_area').h2.text
        except:
            try:
                prod_title = html.h1.find(class_='product_name').text # if desktop page
            except:
                prod_title = ''
        try:
            prod_breadcrumbs = html.find(class_='BreadcrumbsArea').text.replace('\n', ' ')
        except:
            try:
                prod_breadcrumbs = html.find(class_='path').text.replace('\n', ' ') # if desktop page
            except:
                prod_breadcrumbs = ''

    elif 'rakuten' in str(url):
        ec_id = 1007
        try:
            l = re.findall('[\w]+', url)
            prod_id = l[l.index('product')+1]
        except:
            prod_id = ''
        try:
            prod_title = html.find(class_='b-ttl-main').text
        except:
            try:
                prod_title = html.find(class_='product-main-title-text').text # if mobile page
            except:
                prod_title = ''
        try:
            prod_breadcrumbs = html.find(class_='b-breadcrumb').text.replace('\n', ' ')
        except:
            prod_breadcrumbs = ''

    else: #408
        ec_id = 408
        try:
            l = re.findall('[\w]+', p)
            prod_id = l[l.index('Product')+4]
        except:
            prod_id = ''
        try:
            prod_title = html.find(id='Overbig').b.find(class_='Name').text
        except:
            try:
                prod_title = html.find(class_='IN_PP_bk_word_T').div.text # if mobile page
            except:
                prod_title = ''
        try:
            prod_breadcrumbs = html.find(id='map-list').text.replace('\n', ' ')
        except:
            try:
                prod_breadcrumbs = html.find(class_='path').text.replace('\n', ' ') # if mobile page
            except:
                prod_breadcrumbs = ''
                
    result = (url, ec_id, prod_id, prod_title, prod_breadcrumbs)
    print(result)
    return result

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
                    html = await resp.read() #html
                    asyncio.sleep(15)
                    
                    try:
                        html = BeautifulSoup(html, 'html.parser')
                    except Exception as e:
                        print('Error occurred while bs4 parsing html: ', e)
#                         result = (url, 'invalid', 'invalid', 'invalid', 'invalid')
#                         print(result)
#                         results_list.append(result)
                        continue

                    # different EC page structure
                    try:
                        result = parse_rules(html, url)
                    except Exception as e:
#                         result = (url, 'parse error', 'parse error', 'parse error', 'parse error')
                        print('Error occurred while findimg items: ', e)
#                         results_list.append(result)
#                         print('amount crawled: ', len(results_list))
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

#     print(f'Exceed max retry ({url}). Drop task.')
#     result = (url, 'max_retry', 'max_retry', 'max_retry', 'max_retry')
#     print(result)
#     results_list.append(result)
#     print('amount crawled: ', len(results_list))

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
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    date = args.date
    start = args.start

    # 自BigQuery載入 urls
    def query(bigquery_goal):
        t = dd.read_csv(bigquery_goal)
        df = t.compute()
        print('今日user瀏覽產品頁數: ', len(df))
        return df
    
    bigquery_goal = f'gs://tagtoo-bigquery-export/ECAnalysis/new_product_crawler_byDay/CEC_4_{date}'
    print('query from BigQuery: ', bigquery_goal)
    df = query(bigquery_goal)
    
    print('EC_1039 商品頁瀏覽數: ', len(df.query('advertiser_id == [1039]')))
    print('EC_1007 商品頁瀏覽數: ', len(df.query('advertiser_id == [1007]')))
    print('EC_1159 商品頁瀏覽數: ', len(df.query('advertiser_id == [1159]')))
    print('EC_408 商品頁瀏覽數: ', len(df.query('advertiser_id == [408]')))
    allUrls = np.unique(df[~(df['page'].str.contains('rakuten|cyber2.shopping.friday|ecm.momoshop'))].page.values)
#     allUrls = np.unique(df[(df['page'].str.contains('//m.momoshop'))].page.values)
    split_file_num = len(allUrls)/93000

    for i in range(int(start), int(split_file_num+1)):
        print('enter loop and starting with ', str(start), ', should finish at ', split_file_num, 'th round.')
        results_list = list()
        try:
            urls = allUrls[i*93000:(i+1)*93000]
        except Exception as e:
            print(e)
            break
        
        
        print('Getting html files from urls (exclude Rakuten) ...')

        loop = asyncio.get_event_loop()

        future = asyncio.ensure_future(create_connection(args.semaphore, args.connection, urls, results_list))
        loop.run_until_complete(future)

        print('url files finished crawling. Storing to files...')
        print(results_list)
        df_store = pd.DataFrame(results_list)
        df_store = df_store.rename({0:'url', 1:'ec_id', 2:'prod_id', 3:'prod_title', 4:'prod_breadcrumbs'}, axis='columns')
        df_store.to_csv(f'{date}_{i}.csv')

    print('All Job complete.')
    

if __name__ == '__main__':
    main()