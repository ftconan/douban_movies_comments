# douban_movies_comments
douban_movies_comments(豆瓣影评情感分析)
### 豆瓣影评爬虫
1. requests获得爬取最新电影影评ID,NAME列表
2. 伪装登录(HEADER,USER-AGENT,COOKIE)
3. 爬取影评网页
4. beautifulsoup4使用xpath获取评论,解析
5. pandas生成comment.csv,a+模式追加写入数据
6. 循环爬取多页数据,多次写入数据
### TODO 豆瓣影评爬虫 
1. 随机延时爬取豆瓣帐号会被封
2. 没有开启多线程爬取,爬取速度较慢
3. 爬取只能爬取26页,500条,有些有5w,可能需要验证码,无法爬取海量影评
4. douban_movie scrapy搭建爬取项目命令行模式擅未实现
### 豆瓣影评情感分析
1. 数据预处理(去重,去除太短的评论,数据加label,用于评估)
2. 短评词云
3. 构建分类器模型 
* NB:朴素贝叶斯 
* SVC 
* CNN:卷积神经网络 
* RNN:循环神经网络
* GRU
4. 训练和预测
