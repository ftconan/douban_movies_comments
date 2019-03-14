"""
@file:   new_analysis_comment.py
@author: magician
@date:   2018/03/13
"""
import random
from math import ceil

import jieba
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn import metrics

# 停用词
STOPWORDS = pd.read_csv('./data/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'],
                        encoding='utf-8')

# 自动化5种分类器分析
AUTO_FlAG = False

# 构建两层CNN神经网络
learn = tf.contrib.learn
FLAGS = None
# 文档最长长度
MAX_DOCUMENT_LENGTH = 15
# 最小词频数
MIN_WORD_FREQUENCE = 1
# 词嵌入的维度
EMBEDDING_SIZE = 50
# filter个数
N_FILTERS = 10  # 10个神经元
# 感知野大小
WINDOW_SIZE = 20
# filter的形状
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
# 池化
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0


def process_data(csv_name, **kwargs):
    """
    数据预处理(去重,去除太短的评论)
    :param csv_name:     CSV名称
    :param kwargs
    :return:
    """
    try:
        movie_name = kwargs.get('movie_name')
        data_type = kwargs.get('data_type')
        csv_path = './data/' + csv_name + '.csv'
        data_com = pd.read_csv(csv_path)
        data_com.drop_duplicates(inplace=True)
        if data_type == 'train':
            data_com = data_com.drop(['id'], axis=1)
        else:
            data_com = data_com.drop(['user', 'is_watch', 'use_count'], axis=1)
        # mark: like: star >= 3 unlike: star < 3
        data_com['label'] = (data_com.star >= 3) * 1
    except Exception as e:
        return str(e)

    # print(movie_name + 'data:', data_com)

    return data_com


def preprocess_text(content_lines, sentences, category, stopwords):
    """
    文本预处理
    :param content_lines:  评论
    :param sentences:      句子列表
    :param category:       是否喜欢
    :return:
    """
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((' '.join(segs), category))
        except Exception as e:
            # print(line)
            continue


def gru_model(features, target):
    """
    用RNN模型（这里用的是GRU）完成文本分类
    """
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size,sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(features
                                                    , vocab_size=n_words
                                                    , embed_dim=EMBEDDING_SIZE
                                                    , scope='words')
    # Split into list of embedding per word, while removing doc length dim。
    # word_list results to be a list of tensors [batch_size,EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    target = tf.one_hot(target, 15, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


def predict(diff_name, comment, vocab_processor, classifier):
    """
    评论预测
    :param diff_name:  分类器名称
    :param comment:   评论
    :param vocab_processor:   vocab_processor
    :param classifier:        classifier
    :return:
    """
    sentences = []
    preprocess_text([comment], sentences, 'unknown', STOPWORDS)
    x, y = zip(*sentences)
    x_tt = np.array(list(vocab_processor.transform([x[0]])))

    flag = None
    if classifier.predict(x_tt)['class'][0]:
        print(diff_name + ' ' + 'predict: ' + 'like')
        flag = 1
    else:
        print(diff_name + ' ' + 'predict: ' + 'dislike')
        flag = 0

    return flag


def get_comment(data_com):
    """
    获取电影名称
    :param data_com:
    :return:
    """
    movie_name_list = data_com['movie'].tolist()
    name, name_list = movie_name_list[0], []
    for i in range(len(movie_name_list)):
        if i == 0:
            name = movie_name_list[i]
            name_list.append(name)
        else:
            if movie_name_list[i] != name:
                name = movie_name_list[i]
                name_list.append(name)
            else:
                continue

    print('name_list: ', name_list)
    return name_list


def count_favorable_rate():
    """
    好评率统计
    :return:
    """
    # 1.数据预处理(去重,去除太短的评论)
    csv = input('请输入CSV名称:' + '\n') if AUTO_FlAG else 'movie_test_data'
    # movie = input('请输入电影名称:' + '\n') if AUTO_FlAG else '钢铁侠'
    data_com = process_data(csv)

    # 获取电影名称
    movie_name_list = get_comment(data_com)

    # 2.生成训练数据
    print('value_count: ', data_com.label.value_counts())
    data_com_X_1 = data_com[data_com.label == 1]
    data_com_X_0 = data_com[data_com.label == 0]
    like_count = data_com_X_1.label.value_counts()
    dislike_count = data_com_X_0.label.value_counts()
    like_comment = np.random.choice(data_com_X_1['comment'], 1)[0]
    dislike_comment = np.random.choice(data_com_X_0['comment'], 1)[0]
    print('like_comment: ', like_comment)
    print('dislike_comment: ', dislike_comment)

    # 3.下采样
    sentences = []
    preprocess_text(data_com_X_1.comment.dropna().values.tolist(), sentences, 'like', STOPWORDS)
    n = 0
    while n < ceil(like_count / dislike_count):
        preprocess_text(data_com_X_0.comment.dropna().values.tolist(), sentences, 'dislike', STOPWORDS)
        n += 1

    random.shuffle(sentences)

    x, y = zip(*sentences)

    # 4.构建分类器模型(NB,SVC,CNN,RNN,GRU)
    diff_name = 'GRU'
    train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)
    # 处理词汇
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH,
                                                              min_frequency=MIN_WORD_FREQUENCE)
    x_train = np.array(list(vocab_processor.fit_transform(train_data)))
    x_test = np.array(list(vocab_processor.transform(test_data)))
    n_words = len(vocab_processor.vocabulary_)
    print(diff_name + ' ' + 'Total words:%d' % n_words)
    cate_dic = {'like': 1, 'dislike': 0}
    y_train = pd.Series(train_target).apply(lambda x: cate_dic[x], train_target)
    y_test = pd.Series(test_target).apply(lambda x: cate_dic[x], test_target)
    # 5.构建模型
    text_classifier = learn.SKCompat(learn.Estimator(model_fn=gru_model))

    # 6.训练
    text_classifier.fit(x_train, y_train, steps=1000)
    y_predicted = text_classifier.predict(x_test)['class']
    score = metrics.accuracy_score(y_test, y_predicted)
    print(diff_name + ' ' + 'Accuracy:{0:f}'.format(score))

    # 7.预测评论
    predict(diff_name, dislike_comment, vocab_processor, text_classifier)
    predict(diff_name, like_comment, vocab_processor, text_classifier)


if __name__ == '__main__':
    count_favorable_rate()
