"""
@file: crawl_movies
@author: magician
@date: 2018/03/08
"""


import pandas as pd
import random
import time
import requests

from urllib import request
from bs4 import BeautifulSoup as bs
from lxml import etree

BASE_HOT_MOVIES_URL = 'https://movie.douban.com/nowplaying/shanghai/'
MOVIES_COMMENT_URL = 'https://movie.douban.com/subject/{0}/comments'
LAST_PAGE = 100


def get_hot_movie():
    """
    获得爬取最新电影影评ID,NAME列表
    :return:
    """

    # 分析网页函数
    resp = request.urlopen(BASE_HOT_MOVIES_URL)
    html_data = resp.read().decode('utf-8')
    soup = bs(html_data, 'html.parser')
    hot_movies = soup.find_all('div', id='nowplaying')
    hot_movies_list = hot_movies[0].find_all('li', class_='list-item')
    hot_list = []
    for item in hot_movies_list:
        hot_dict = dict()
        hot_dict['id'] = item['data-subject']
        for tag_img_item in item.find_all('img'):
            hot_dict['name'] = tag_img_item['alt']
            hot_list.append(hot_dict)

    return hot_list


def mask_login(movie_id):
    """
    伪装登录
    :param movie_id 电影ID
    :return:
    """
    user_agent = [
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2226.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.4; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2224.3 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 4.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.3319.102 Safari/537.36",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.2309.372 Safari/537.36",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.2117.157 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1866.237 Safari/537.36"
    ]
    header = {
        'User-Agent': random.choice(user_agent),
        'Host': 'movie.douban.com',
        'Referer': 'https://movie.douban.com/subject/{0}/?from=showing'.format(movie_id)
    }

    # session = requests.Session()

    cookie = {
        'cookie': 'bid=MPqc-GreJ1Q; douban-fav-remind=1; __utmc=30149280; __utmz=30149280.1551839465.3.3.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; push_noty_num=0; push_doumail_num=0; __utmc=223695111; _vwo_uuid_v2=D92011475AFC39824B3BCDD9B8F455BDD|b3fa3272efd82a0ef8a4353e7a067670; douban-profile-remind=1; ll="108296"; ps=y; _pk_ses.100001.4cf6=*; ap_v=0,6.0; __utma=30149280.864854032.1540641565.1552186724.1552219239.8; __utmt=1; dbcl2="164706329:xMClm0EBv+4"; ck=e4Yw; __utma=223695111.489329365.1551946716.1552188527.1552219527.6; __utmb=223695111.0.10.1552219527; __utmz=223695111.1552219527.6.2.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/passport/login; __utmv=30149280.16470; loc-last-index-location-id="108296"; __utmb=30149280.7.10.1552219239; _pk_id.100001.4cf6=bac874e7fea6b4b1.1551946712.6.1552219802.1552189422.'
    }

    mask_user = {
        'header': header,
        'cookie': cookie
    }

    return mask_user


def get_comment(url, **kwargs):
    """
    爬取影评网页
    :param
    url:    网页url
    kwargs:
           header: 请求头
           cookie  Cookie
    :return:
    """
    header = kwargs.get('header')
    cookie = kwargs.get('cookie')
    timeout = 3

    time.sleep(random.randint(5, 15))
    # 爬取影评网页(原始数据)
    response = requests.get(url, headers=header, cookies=cookie, timeout=timeout)
    if response.status_code != 200:
        print('status_code:', response.status_code, 'response:', response)
        return response.status_code

    return response


def analysis_comment(movie_name, comment):
    """
    分析影评,获得想要得内容
    :param movie_name: 电影名称
    :param comment:    影评网页源数据
    :return:
    """
    comment_list = []
    user = comment.xpath("./h3/span[@class='comment-info']/a/text()")[0]  # 用户
    watched = comment.xpath("./h3/span[@class='comment-info']/span[1]/text()")[0]  # 是否看过
    rating = comment.xpath("./h3/span[@class='comment-info']/span[2]/@title")  # 五星评分
    if len(rating) > 0:
        rating = rating[0]

    comment_time = comment.xpath("./h3/span[@class='comment-info']/span[3]/@title")  # 评论时间
    if len(comment_time) > 0:
        comment_time = comment_time[0]
    else:
        # 有些评论是没有五星评分, 需赋空值
        comment_time = rating
        rating = ''

    votes = comment.xpath("./h3/span[@class='comment-vote']/span/text()")[0]  # "有用"数
    content = comment.xpath("./p/span[@class='short']/text()")[0]  # 评论内容

    comment_list.append(movie_name)
    comment_list.append(user)
    comment_list.append(watched)
    comment_list.append(rating)
    comment_list.append(comment_time)
    comment_list.append(votes)
    comment_list.append(content.strip())

    return comment_list


def save_comment(first_flag, comment_data):
    """
    保存影评
    :param first_flag:     flag
    :param comment_data:   data
    :return:
    """
    # 写入csv文件,'a+'是追加模式
    try:
        if first_flag:
            csv_headers = ['电影', '用户', '是否看过', '五星评分', '评论时间', '有用数', '评论内容']
            comment_data.to_csv('./data/comment.csv', header=csv_headers, index=False, mode='a+', encoding='utf-8')
        else:
            comment_data.to_csv('./data/comment.csv', header=False, index=False, mode='a+', encoding='utf-8')
    except UnicodeEncodeError:
        print("编码错误, 该数据无法写到文件中, 直接忽略该数据")

    return True


def start_spiders():
    """
    爬虫启动主程序
    :return:
    """
    # 1.获得爬取最新电影影评ID,NAME列表
    first_flag = False
    hot_movies = get_hot_movie()
    print('hot_movies:', hot_movies, 'length:', len(hot_movies))

    for movie in hot_movies:
        print('movie'+ str(hot_movies.index(movie) + 1) + 'crawl:', movie)

        movie_id = movie.get('id', '')
        movie_name = movie.get('name', '')
        base_url = MOVIES_COMMENT_URL.format(movie_id)
        start_url = base_url + '?start={0}'.format(str(0))

        # 2.伪装登录
        user = mask_login(movie_id)

        # 3.爬取影评网页
        response = get_comment(start_url, **user)

        page_count = 1
        while response.status_code == 200:
            # 爬取100页
            print(movie_name + '第' + str(page_count+1) + 'crawl......')
            if page_count > 100:
                break

            selector = etree.HTML(response.text)
            next_page = selector.xpath("//div[@id='paginator']/a[@class='next']/@href")
            try:
                next_page = next_page[0]
            except IndexError:
                break

            next_url = base_url + next_page

            # 获取评论,解析
            origin_comments = selector.xpath("//div[@class='comment']")
            comment_data = []
            for comment in origin_comments:
                # 4.分析影评,获得想要得内容
                comment_data.append(analysis_comment(movie_name, comment))

            # 5.生成csv
            data = pd.DataFrame(comment_data)
            save_comment(first_flag, data)

            data = []

            # 爬取下一页
            response = get_comment(next_url, **user)
            page_count += 1
            first_flag = False

    return True


if __name__ == '__main__':
    start_spiders()
