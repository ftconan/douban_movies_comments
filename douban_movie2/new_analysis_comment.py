"""
@file:   new_analysis_comment.py
@author: magician
@date:   2018/03/13
"""
import os
import random
import jieba
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from tensorflow.contrib.layers.python.layers import encoders
from wordcloud import WordCloud

# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 停用词
STOPWORDS = pd.read_csv('./data/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'],
                        encoding='utf-8')

# 自动化5种分类器分析
AUTO_FlAG = False

# 构建两层CNN神经网络
learn = tf.contrib.learn
FLAGS = None
# 文档最长长度
MAX_DOCUMENT_LENGTH = 100
# 最小词频数
MIN_WORD_FREQUENCE = 2
# 词嵌入的维度
EMBEDDING_SIZE = 20
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
            data_com = data_com.drop(['id'], axis=1)
            data_com = data_com.drop(['user', 'is_watch', 'use_count'], axis=1)
        # mark: like: star >= 3 unlike: star < 3
        data_com['label'] = (data_com.star >= 3) * 1
    except Exception as e:
        return str(e)

    # print(movie_name + 'data:', data_com)

    return data_com


def generate_word_cloud(data, stopwords, **kwargs):
    """
    生成词云
    :param data        词云数据
    :param stopwords   停用词
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


class TextClassifier:
    """
    文本分类器
    """
    def __init__(self, classifier, vectorizer):
        """
        init
        :param classifier:
        :param vectorizer:
        """
        self.classifier = classifier
        self.vectorizer = vectorizer

    def features(self, X):
        """
        特征
        :param X:
        :return:
        """
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        """
        训练
        :param X:
        :param y:
        :return:
        """
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        """
        预测
        :param x:
        :return:
        """
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        """
        得分
        :param X:
        :param y:
        :return:
        """
        return self.classifier.score(self.features(X), y)


def cnn_model(features, target):
    """
    2层的卷积神经网络，用于短文本分类
    1.先把词转成词嵌入
    2.我们得到一个形状为[n_words, EMBEDDING_SIZE]的词表映射矩阵
    3.接着我们可以把一批文本映射成[batch_size, sequence_length, EMBEDDING_SIZE]的矩阵形式
    :param features:
    :param target:
    :return:
    """
    # 对词编码
    target = tf.one_hot(target, 15, 1, 0)
    word_vectors = tf.contrib.layers.embed_sequence(features,
                                                    vocab_size=n_words,
                                                    embed_dim=EMBEDDING_SIZE,
                                                    scope='words')
    word_vectors = tf.expand_dims(word_vectors, 3)

    with tf.variable_scope('CNN_Layer1'):
        # 添加卷积层做滤波
        conv1 = tf.contrib.layers.convolution2d(word_vectors,
                                                N_FILTERS,
                                                FILTER_SHAPE1,
                                                padding='VALID')
        # 添加RELU非线性
        conv1 = tf.nn.relu(conv1)
        # 最大池化
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, POOLING_WINDOW, 1, 1],
                               strides=[1, POOLING_STRIDE, 1, 1],
                               padding='SAME')
        # 对矩阵进行转置，以满足形状
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])

    with tf.variable_scope('CNN_Layer2'):
        # 第二卷积层
        conv2 = tf.contrib.layers.convolution2d(pool1,
                                                N_FILTERS,
                                                FILTER_SHAPE2,
                                                padding='VALID')
        # 抽取特征
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # 全连接层
    logits = tf.contrib.layers.fully_connected(pool2, 15, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    # 优化器
    train_op = tf.contrib.layers.optimize_loss(loss,
                                               tf.contrib.framework.get_global_step(),
                                               optimizer='Adam',
                                               learning_rate=0.01)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


def bag_of_words_model(features, target):
    """
    先转成词袋模型
    :param features:
    :param target:
    :return:
    """
    target = tf.one_hot(target, 15, 1, 0)
    features = encoders.bow_encoder(features,
                                    vocab_size=n_words,
                                    embed_dim=EMBEDDING_SIZE)
    logits = tf.contrib.layers.fully_connected(features,
                                               15,
                                               activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    train_op = tf.contrib.layers.optimize_loss(loss,
                                               tf.contrib.framework.get_global_step(),
                                               optimizer='Adam',
                                               learning_rate=0.01)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


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


def predict(diff_name, commment, vocab_processor):
    """
    评论预测
    :param diff_name:  分类器名称
    :param commment:   评论
    :param vocab_processor:   vocab_processor
    :return:
    """
    sentences = []
    preprocess_text([commment], sentences, 'unknown', STOPWORDS)
    x, y = zip(*sentences)
    x_tt = np.array(list(vocab_processor.transform([x[0]])))

    if classifier.predict(x_tt)['class'][0]:
        print(diff_name+'predict: '+'like')
    else:
        print(diff_name+'predict: '+'dislike')

    return True


if __name__ == '__main__':
    # 1.数据预处理(去重,去除太短的评论)
    csv = input('请输入CSV名称:'+'\n') if AUTO_FlAG else 'movie_train_data'
    movie = input('请输入电影名称:'+'\n') if AUTO_FlAG else '钢铁侠'
    data_com = process_data(csv, movie)

    # 2.短评词云
    # data_type = input('请输入数据类型(train or test):'+'\n')
    # generate_word_cloud(data_com, **{'data_type': data_type})

    # 3.生成训练数据
    print('value_count: ', data_com.label.value_counts())
    data_com_X_1 = data_com[data_com.label == 1]
    data_com_X_0 = data_com[data_com.label == 0]
    like_comment = np.random.choice(data_com_X_1['comment'], 1)[0]
    dislike_comment = np.random.choice(data_com_X_0['comment'], 1)[0]
    print('like_comment: ', like_comment)
    print('dislike_comment: ', dislike_comment)

    # 下采样
    sentences = []
    preprocess_text(data_com_X_1.comment.dropna().values.tolist(), sentences, 'like', STOPWORDS)
    n = 0
    while n < 41:
        preprocess_text(data_com_X_0.comment.dropna().values.tolist(), sentences, 'dislike', STOPWORDS)
        n += 1

    random.shuffle(sentences)
    # for sentence in sentences[:2]:
    #     print(sentence[0], sentence[1])

    x, y = zip(*sentences)

    # 4.构建分类器模型(NB,SVC,CNN,RNN,GRU)
    diff_name_list = ['NB', 'SVC', 'CNN', 'RNN', 'GRU']
    classifier_name = input('请输入数据类型(NB OR SVC OR CNN OR RNN OR GRU OR Q):'+'\n') if AUTO_FlAG else diff_name_list
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
    text_classifier = None

    for diff_name in diff_name_list:
        if diff_name in ['NB', 'SVC']:
            if diff_name == 'NB':
                classifier = MultinomialNB()
                vectorizer = CountVectorizer(analyzer='word'
                                             , ngram_range=(1, 4)
                                             , max_features=20000)
                text_classifier = TextClassifier(classifier, vectorizer)

            else:
                classifier = SVC(kernel='linear')
                vectorizer = TfidfVectorizer(analyzer='word',
                                             ngram_range=(1, 4),
                                             max_features=20000)
                text_classifier = TextClassifier(classifier, vectorizer)

            text_classifier.fit(x_train, y_train)
            print(diff_name + ' ' + 'score', text_classifier.score(x_test, y_test))

            # 预测评论
            print(diff_name+' '+'predict', text_classifier.predict(like_comment))
            print(diff_name+' '+'predict', text_classifier.predict(dislike_comment))
        elif diff_name in ['CNN', 'RNN', 'GRU']:
            train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)
            # 处理词汇
            vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH,
                                                                      min_frequency=MIN_WORD_FREQUENCE)
            x_train = np.array(list(vocab_processor.fit_transform(train_data)))
            x_test = np.array(list(vocab_processor.transform(test_data)))
            n_words = len(vocab_processor.vocabulary_)
            print(diff_name + ' ' + 'Total words:%d' % n_words)
            if diff_name == 'CNN':
                cate_dic = {'like': 1, 'dislike': 0}
                y_train = pd.Series(train_target).apply(lambda x: cate_dic[x], train_target)
                y_test = pd.Series(test_target).apply(lambda x: cate_dic[x], test_target)
                # 4.构建模型
                text_classifier = learn.SKCompat(learn.Estimator(model_fn=cnn_model))
            elif diff_name == 'RNN':
                # 使用RNN完成文本分类
                learn = tf.contrib.learn
                FLAGS = None
                MAX_DOCUMENT_LENGTH = 15
                MIN_WORD_FREQUENCE = 1
                EMBEDDING_SIZE = 50

                text_classifier = learn.SKCompat(learn.Estimator(model_fn=bag_of_words_model))
            else:
                text_classifier = learn.SKCompat(learn.Estimator(model_fn=gru_model))

            # 训练
            text_classifier.fit(x_train, y_train, steps=1000)
            y_predicted = text_classifier.predict(x_test)['class']
            score = metrics.accuracy_score(y_test, y_predicted)
            print(diff_name+' '+'Accuracy:{0:f}'.format(score))

            # 预测评论
            # if diff_name in ['CNN','RNN', 'GRU']:
            #     predict(diff_name, dislike_comment, vocab_processor)
            #     predict(diff_name, like_comment, vocab_processor)
        else:
            print('请输入正确分类器类型')
