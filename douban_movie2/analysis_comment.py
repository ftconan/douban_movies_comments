"""
@file:   analysis_comment.py
@author: magician
@date:   2018/03/11
"""
import warnings
import random
import jieba
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from tensorflow.contrib.layers.python.layers import encoders
from wordcloud import WordCloud
from douban_movie2.crawl_movies import get_hot_movie


warnings.filterwarnings("ignore")
mpl.rcParams['figure.figsize']=(10.0,5.0)

data_com = pd.read_csv('./data/comment.csv')

# 1.数据预处理(去重,去除太短的评论)
data_com.drop_duplicates(inplace=True)
data_com = data_com.drop(['user', 'is_watch', 'use_count'], axis=1)
# data_com['comment'] = data_com['comment'].apply(lambda x: len(str(x)) > 5)
# mark: like: star >= 3 unlike: star < 3
data_com['label'] = (data_com.star >= 3) * 1
# print(data_com.info())
# print(data_com.head(2))

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
# print(data_com_X)

# small comment
mpl.rc('figure', figsize=(14, 7))
mpl.rc('font', size=14)
mpl.rc('axes', grid=False)
mpl.rc('axes', facecolor='white')
sns.distplot(data_com_X.comment_time.apply(lambda x: int(x[0:4]) + int(x[5:7]))
             , bins=100, kde=False, rug=True)
plt.xlabel('time')
plt.ylabel('Number of short_comment')
# plt.show()

# 2.短评词云
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
stopwords = pd.read_csv('./data/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'], encoding='utf-8')
words_df = words_df[~words_df.segment.isin(stopwords.stopword)]
# 统计词频
words_stat = words_df.groupby(by=['segment'])['segment'].agg({'计数': np.size})
words_stat = words_stat.reset_index().sort_values(by=['计数'], ascending=False)
# print(words_stat.head())

# 词云
# word_cloud = WordCloud(font_path='./data/simhei.ttf', background_color='white', max_font_size=80)
# words_frequence = {x[0]:x[1] for x in words_stat.head(1000).values}
# print(words_frequence)
# word_cloud = word_cloud.fit_words(words_frequence)
# plt.imshow(word_cloud)

# 3.根据短评文本和点评人是否喜欢作为训练数据，构建情感褒贬分析分类器
print(data_com_X.label.value_counts())


def preprocess_text(content_lines, sentences, category):
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


data_com_X_1 = data_com_X[data_com_X.label == 1]
data_com_X_0 = data_com_X[data_com_X.label == 0]

# 生成训练数据
sentences = []
preprocess_text(data_com_X_1.comment.dropna().values.tolist(), sentences, 'like')
n = 0
while n < 3:
    preprocess_text(data_com_X_0.comment.dropna().values.tolist(), sentences, 'nlike')
    n += 1

random.shuffle(sentences)
for sentence in sentences[:2]:
    print(sentence[0], sentence[1])

x,y = zip(*sentences)


# NB(朴素贝叶斯)
vec = CountVectorizer(
    analyzer='word',  # tokenise by character ngrams
    ngram_range=(1, 4),  # use ngrams of size 1 and 2
    max_features=20000,  # keep the most common 1000 ngrams
)


def stratifiedkfold_cv(x,y,clf_class,shuffle=True,n_folds=5,**kwargs):
    """
    stratifiedkfold_cv
    :param x:
    :param y:
    :param clf_class:
    :param shuffle:
    :param n_folds:
    :param kwargs:
    :return:
    """
    stratifiedk_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y[:]
    for train_index, test_index in stratifiedk_fold:
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)

    return y_pred


class TextClassifier:
    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word'
                                          , ngram_range=(1, 4)
                                          , max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


text_classifier = TextClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
text_classifier.fit(x_train, y_train)
print('NB predict', text_classifier.predict('准确来说应该叫《复联：缘起》或者《一家Marvel的“诞生”》。一点幽默，一点情怀，一点特效，剩下的'
                                            '七点送给A爆的惊队本队！不过这挂开的是不是有点过分…从天而降，居然有一种致敬《功夫》的幻觉。'))
print('NB predict', text_classifier.predict('比神奇女侠差远了，不是在说颜值'))
print('NB score', text_classifier.score(x_test, y_test))


# 用SVC完成中文文本分类器
class TextClassifier:
    """
    SVC
    """
    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier
        self.vectorizer = TfidfVectorizer(analyzer='word',
                                          ngram_range=(1,4),
                                          max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


text_classifier=TextClassifier()
text_classifier.fit(x_train,y_train)
print('SVC predict', text_classifier.predict('一点 不觉得震撼'))
print('SVC predict', text_classifier.predict('好看'))
print('SVC', text_classifier.score(x_test,y_test))


# 用CNN做中文文本分类,数据预处理
# stopwords = pd.read_csv('./data/stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'], encoding='utf-8')
# stopwords = stopwords.values

#
#
# # 构建两层CNN神经网络
# learn = tf.contrib.learn
# FLAGS = None
# # 文档最长长度
# MAX_DOCUMENT_LENGTH = 100
# # 最小词频数
# MIN_WORD_FREQUENCE = 2
# # 词嵌入的维度
# EMBEDDING_SIZE = 20
# # filter个数
# N_FILTERS = 10  # 10个神经元
# # 感知野大小
# WINDOW_SIZE = 20
# # filter的形状
# FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
# FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
# # 池化
# POOLING_WINDOW = 4
# POOLING_STRIDE = 2
# n_words = 0
#
#
# def cnn_model(features, target):
#     """
#     2层的卷积神经网络，用于短文本分类
#     1.先把词转成词嵌入
#     2.我们得到一个形状为[n_words, EMBEDDING_SIZE]的词表映射矩阵
#     3.接着我们可以把一批文本映射成[batch_size, sequence_length, EMBEDDING_SIZE]的矩阵形式
#     :param features:
#     :param target:
#     :return:
#     """
#     # 对词编码
#     target = tf.one_hot(target, 15, 1, 0)
#     word_vectors = tf.contrib.layers.embed_sequence(features,
#                                                     vocab_size=n_words,
#                                                     embed_dim=EMBEDDING_SIZE,
#                                                     scope='words')
#     word_vectors = tf.expand_dims(word_vectors, 3)
#
#     with tf.variable_scope('CNN_Layer1'):
#         # 添加卷积层做滤波
#         conv1 = tf.contrib.layers.convolution2d(word_vectors,
#                                                 N_FILTERS,
#                                                 FILTER_SHAPE1,
#                                                 padding='VALID')
#         # 添加RELU非线性
#         conv1 = tf.nn.relu(conv1)
#         # 最大池化
#         pool1 = tf.nn.max_pool(conv1,
#                                ksize=[1, POOLING_WINDOW, 1, 1],
#                                strides=[1, POOLING_STRIDE, 1, 1],
#                                padding='SAME')
#         # 对矩阵进行转置，以满足形状
#         pool1 = tf.transpose(pool1, [0, 1, 3, 2])
#
#     with tf.variable_scope('CNN_Layer2'):
#         # 第二卷积层
#         conv2 = tf.contrib.layers.convolution2d(pool1,
#                                                 N_FILTERS,
#                                                 FILTER_SHAPE1,
#                                                 padding='VALID')
#         # 抽取特征
#         pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
#
#     # 全连接层
#     logits = tf.contrib.layers.fully_connected(pool2, 15, activation_fn=None)
#     loss = tf.losses.softmax_cross_entropy(target, logits)
#     # 优化器
#     train_op = tf.contrib.layers.optimize_loss(loss,
#                                                tf.contrib.framework.get_global_step(),
#                                                optimizer='Adam',
#                                                learning_rate=0.01)
#
#     return ({
#                 'class': tf.argmax(logits, 1),
#                 'prob': tf.nn.softmax(logits)
#             }, loss, train_op)


# 处理词汇
# vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
# x_train = np.array(list(vocab_processor.fit_transform(train_data)))
# x_test = np.array(list(vocab_processor.transform(test_data)))
# n_words = len(vocab_processor.vocabulary_)
# print('Total words:%d' % n_words)
#
# cate_dic = {'like': 1, 'nlike': 0}
# y_train = pd.Series(train_target).apply(lambda x:cate_dic[x], train_target)
# y_test = pd.Series(test_target).apply(lambda x:cate_dic[x], test_target)

# 4.构建模型
# classifier = learn.SKCompat(learn.Estimator(model_fn=cnn_model))

# 训练和预测
# classifier.fit(x_train, y_train, steps=1000)
# y_predicted = classifier.predict(x_test)['class']
# score = metrics.accuracy_score(y_test, y_predicted)
# print('Accuracy:{0:f}'.format(score))

# # 使用RNN完成文本分类
# learn = tf.contrib.learn
# FLAGS = None
# MAX_DOCUMENT_LENGTH = 15
# MIN_WORD_FREQUENCE = 1

# 处理词汇
# vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH,
#                                                           min_frequency=MIN_WORD_FREQUENCE)
# x_train = np.array(list(vocab_processor.fit_transform(train_data)))
# x_test = np.array(list(vocab_processor.transform(test_data)))
# print('Total words: %d' % n_words)
#
#
# def bag_of_words_model(features, target):
#     """
#     先转成词袋模型
#     :param features:
#     :param target:
#     :return:
#     """
#     target = tf.one_hot(target, 15, 1, 0)
#     features = encoders.bow_encoder(features,
#                                     vocab_size=n_words,
#                                     embed_dim=EMBEDDING_SIZE)
#     logits = tf.contrib.layers.fully_connected(features,
#                                                15,
#                                                activation_fn=None)
#     loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
#     train_op = tf.contrib.layers.optimize_loss(loss,
#                                                tf.contrib.framework.get_global_step(),
#                                                optimizer='Adam',
#                                                learning_rate=0.01)
#
#     return ({
#                 'class': tf.argmax(logits, 1),
#                 'prob': tf.nn.softmax(logits)
#             }, loss, train_op)

#
# model_fn = bag_of_words_model
# classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))
#
# # 5.测试和预测
# classifier.fit(x_train, y_train, steps=1000)
# y_predicted = classifier.predict(x_test)['class']
# score = metrics.accuracy_score(y_test, y_predicted)
# print('Accuracy: {0:f}'.format(score))

# def rnn_model(features, target):
#     """
#     用RNN模型（这里用的是GRU）完成文本分类
#     """
#     # Convert indexes of words into embeddings.
#     # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
#     # maps word indexes of the sequence into [batch_size,sequence_length,
#     # EMBEDDING_SIZE].
#     word_vectors = tf.contrib.layers.embed_sequence(features
#                                                     , vocab_size=n_words
#                                                     , embed_dim=EMBEDDING_SIZE
#                                                     , scope='words')
#     # Split into list of embedding per word, while removing doc length dim。
#     # word_list results to be a list of tensors [batch_size,EMBEDDING_SIZE].
#     word_list = tf.unstack(word_vectors, axis=1)
#
#     # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
#     cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)
#
#     # Create an unrolled Recurrent Neural Networks to length of
#     # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
#     _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
#
#     # Given encoding of RNN, take encoding of last step (e.g hidden size of the
#     # neural network of last step) and pass it as features for logistic
#     # regression over output classes.
#     target = tf.one_hot(target, 15, 1, 0)
#     logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
#     loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
#
#     # Create a training op.
#     train_op = tf.contrib.layers.optimize_loss(
#         loss,
#         tf.contrib.framework.get_global_step(),
#         optimizer='Adam',
#         learning_rate=0.01)
#
#     return ({
#                 'class': tf.argmax(logits, 1),
#                 'prob': tf.nn.softmax(logits)
#             }, loss, train_op)

#
# model_fn = rnn_model
# classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))
#
# # Train and predict
# classifier.fit(x_train, y_train, steps=1000)
# y_predicted = classifier.predict(x_test)['class']
# score = metrics.accuracy_score(y_test, y_predicted)
# print('Accuracy:{0:f}'.format(score))
