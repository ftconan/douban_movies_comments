"""
@file:   new_analysis_comment.py
@author: magician
@date:   2018/03/13
"""
import jieba
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def process_data(csv_name, movie_name, **kwargs):
    """
    数据预处理(去重,去除太短的评论)
    :param csv_name:     CSV名称
    :param movie_name:   电影名称
    :param kwargs
    :return:
    """
    try:
        data_type = kwargs.get('data_type')
        csv_path = './data/' + csv_name + '.csv'
        data_com = pd.read_csv(csv_path)
        data_com.drop_duplicates(inplace=True)
        if data_type == 'train':
            data_com = data_com.drop(['user', 'is_watch', 'use_count'], axis=1)
        else:
            data_com = data_com.drop(['id'], axis=1)
        # mark: like: star >= 3 unlike: star < 3
        data_com['label'] = (data_com.star >= 3) * 1
    except Exception as e:
        return str(e)

    print(movie_name + 'data:', data_com)

    return data_com


def generate_word_cloud(data, **kwargs):
    """
    生成词云
    :param data 词云数据
    :param kwargs
    :return:
    """
    movie_name = kwargs.get('movie_name')
    if movie_name:
        data_com_X = data[data.movie == movie_name]
    else:
        data_com_X = data

    content_X = data_com_X.comment.dropna().values.tolist()
    # 导入,分词
    segment = []
    for line in content_X:
        try:
            segs = jieba.lcut(line)
            for seg in segs:
                if len(seg) > 1 and seg != '\r\n':
                    segment.append(seg)
        except Exception as e:
            # print(line)
            continue

    # 去停用词
    words_df = pd.DataFrame({'segment': segment})
    stopwords = pd.read_csv('./data/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'],
                            encoding='utf-8')
    words_df = words_df[~words_df.segment.isin(stopwords.stopword)]
    # 统计词频
    words_stat = words_df.groupby(by=['segment'])['segment'].agg({'计数': np.size})
    words_stat = words_stat.reset_index().sort_values(by=['计数'], ascending=False)
    # print(words_stat.head())

    # 词云
    word_cloud = WordCloud(font_path='./data/simhei.ttf', background_color='white', max_font_size=80)
    words_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
    print(words_frequence)
    word_cloud = word_cloud.fit_words(words_frequence)
    plt.imshow(word_cloud)

    return True


if __name__ == '__main__':
    # 1.数据预处理(去重,去除太短的评论)
    csv = input('请输入CSV名称:'+'\n')
    movie = input('请输入电影名称:'+'\n')
    data_com = process_data(csv, movie)
    # 2.短评词云
    data_type = input('请输入数据类型(train or test):'+'\n')
    generate_word_cloud(data_com, **{'data_type': data_type})
    # 3.构建分类器模型(NB,SVC,CNN,RNN,GRU)
    # 4.训练和预测
    # input('请')
    pass
