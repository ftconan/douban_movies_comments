"""
@file:   pre_comment.py
@author: magician
@date:   2018/03/11
"""
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from douban_movie2.crawl_movies import get_hot_movie

data_com = pd.read_csv('./data/comment.csv')

# 1.数据预处理(去重,去除太短的评论)
data_com.drop_duplicates(inplace=True)
data_com = data_com.drop(['user', 'is_watch', 'use_count'], axis=1)
data_com['comment'] = data_com['comment'].apply(lambda x: len(str(x)) > 5)
# mark: like: star >= 3 unlike: star < 3
data_com['label'] = (data_com.star >= 3) * 1
print(data_com.info())
print(data_com.head(2))

# 电影短评论数量分布(总体分布可见下图，横轴是每个电影的短评量，纵轴是相应的电影个数)
data_com.movie.value_counts().hist(bins=20)
plt.ylabel('Number of movie')
plt.xlabel('Number of short_comment of each movie')
# plt.show()

# 拉取电影
# hot_movie = get_hot_movie()
# hot_movie_name = [item['name'] for item in hot_movie]
# print(hot_movie_name)

# 惊奇队长
data_com_X = data_com[data_com.movie == '惊奇队长']
print('爬取《惊奇队长》的短评数：', data_com_X.shape[0])

mpl.rc('figure', figsize = (14, 7))
mpl.rc('font', size = 14)
mpl.rc('axes', grid = False)
mpl.rc('axes', facecolor = 'white')
sns.distplot(data_com_X.comment_time.apply(lambda x: int(x.year)+float(x.month/12.0))
             , bins=100, kde=False, rug=True)
plt.xlabel('time')
plt.ylabel('Number of short_comment')
