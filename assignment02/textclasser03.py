# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:35:35 2020

@author: 罗福杰
"""

from sklearn.datasets import  fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

'''
文本特征提取：
    将文本数据转化成特征向量的过程
    比较常用的文本特征表示法为词袋法
词袋法：
    不考虑词语出现的顺序，每个出现过的词汇单独作为一列特征
    这些不重复的特征词汇集合为词表
    每一个文本都可以在很长的词表上统计出一个很多列的特征向量
    如果每个文本都出现的词汇，一般被标记为 停用词 不计入特征向量
    
主要有两个api来实现 CountVectorizer 和 TfidfVectorizer
CountVectorizer：
    只考虑词汇在文本中出现的频率
TfidfVectorizer：
    除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量
    能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征

相比之下，文本条目越多，Tfid的效果会越显著


下面对两种提取特征的方法，分别设置停用词和不停用，
使用朴素贝叶斯，决策树，K近邻进行分类预测，比较评估效果

'''


# 1 下载新闻数据
news = fetch_20newsgroups(subset="all")


# 2 分割训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.20,
                                                    random_state=30)


# 3.1 采用普通统计CountVectorizer提取特征向量
# 默认配置不去除停用词
count_vec = CountVectorizer()
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)
# 去除停用词
count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_stop_train = count_stop_vec.fit_transform(x_train)
x_count_stop_test = count_stop_vec.transform(x_test)

# 3.2 采用TfidfVectorizer提取文本特征向量
# 默认配置不去除停用词
tfid_vec = TfidfVectorizer()
x_tfid_train = tfid_vec.fit_transform(x_train)
x_tfid_test = tfid_vec.transform(x_test)
# 去除停用词
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
x_tfid_stop_train = tfid_stop_vec.fit_transform(x_train)
x_tfid_stop_test = tfid_stop_vec.transform(x_test)


# 4 使用朴素贝叶斯分类器  分别对两种提取出来的特征值进行学习和预测
# 对普通通统计CountVectorizer提取特征向量 学习和预测
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)   # 学习
mnb_count_y_predict = mnb_count.predict(x_count_test)   # 预测
# 去除停用词
mnb_count_stop = MultinomialNB()
mnb_count_stop.fit(x_count_stop_train, y_train)   # 学习
mnb_count_stop_y_predict = mnb_count_stop.predict(x_count_stop_test)    # 预测

# 对TfidfVectorizer提取文本特征向量 学习和预测
mnb_tfid = MultinomialNB()
mnb_tfid.fit(x_tfid_train, y_train)
mnb_tfid_y_predict = mnb_tfid.predict(x_tfid_test)
# 去除停用词
mnb_tfid_stop = MultinomialNB()
mnb_tfid_stop.fit(x_tfid_stop_train, y_train)   # 学习
mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(x_tfid_stop_test)  # 预测

# 5 模型评估
# 对普通统计CountVectorizer提取的特征学习模型进行评估
print("未去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count.score(x_count_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_count_y_predict, y_test))
print("去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count_stop.score(x_count_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_count_stop_y_predict, y_test))

# 对TfidVectorizer提取的特征学习模型进行评估
print("TfidVectorizer提取的特征学习模型准确率：", mnb_tfid.score(x_tfid_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_tfid_y_predict, y_test))
print("去除停用词的TfidVectorizer提取的特征学习模型准确率：", mnb_tfid_stop.score(x_tfid_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_tfid_stop_y_predict, y_test))




# 下面是决策树算法：
# 4 使用决策树分类器  分别对两种提取出来的特征值进行学习和预测
# 对普通通统计CountVectorizer提取特征向量 学习和预测
dt_count = DecisionTreeClassifier()
dt_count.fit(x_count_train, y_train)   # 学习
dt_count_y_predict = dt_count.predict(x_count_test)   # 预测
# 去除停用词
dt_count_stop = DecisionTreeClassifier()
dt_count_stop.fit(x_count_stop_train, y_train)   # 学习
dt_count_stop_y_predict = dt_count_stop.predict(x_count_stop_test)    # 预测

# 对TfidfVectorizer提取文本特征向量 学习和预测
dt_tfid = DecisionTreeClassifier()
dt_tfid.fit(x_tfid_train, y_train)
dt_tfid_y_predict = dt_tfid.predict(x_tfid_test)
# 去除停用词
dt_tfid_stop = DecisionTreeClassifier()
dt_tfid_stop.fit(x_tfid_stop_train, y_train)   # 学习
dt_tfid_stop_y_predict = dt_tfid_stop.predict(x_tfid_stop_test)  # 预测

# 5 模型评估
print("--------------决策树的结果--------------")
# 对普通统计CountVectorizer提取的特征学习模型进行评估
print("未去除停用词的CountVectorizer提取的特征学习模型准确率：", dt_count.score(x_count_test, y_test))
print("更加详细的评估指标:\n", classification_report(dt_count_y_predict, y_test))
print("去除停用词的CountVectorizer提取的特征学习模型准确率：", dt_count_stop.score(x_count_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(dt_count_stop_y_predict, y_test))

# 对TfidVectorizer提取的特征学习模型进行评估
print("TfidVectorizer提取的特征学习模型准确率：", dt_tfid.score(x_tfid_test, y_test))
print("更加详细的评估指标:\n", classification_report(dt_tfid_y_predict, y_test))
print("去除停用词的TfidVectorizer提取的特征学习模型准确率：", dt_tfid_stop.score(x_tfid_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(dt_tfid_stop_y_predict, y_test))



# 下面是KNN算法：
# 4 使用决策树分类器  分别对两种提取出来的特征值进行学习和预测
# 对普通通统计CountVectorizer提取特征向量 学习和预测
knn_count = KNeighborsClassifier()
knn_count.fit(x_count_train, y_train)   # 学习
knn_count_y_predict = knn_count.predict(x_count_test)   # 预测
# 去除停用词
knn_count_stop = KNeighborsClassifier()
knn_count_stop.fit(x_count_stop_train, y_train)   # 学习
knn_count_stop_y_predict = knn_count_stop.predict(x_count_stop_test)    # 预测

# 对TfidfVectorizer提取文本特征向量 学习和预测
knn_tfid = KNeighborsClassifier()
knn_tfid.fit(x_tfid_train, y_train)
knn_tfid_y_predict = knn_tfid.predict(x_tfid_test)
# 去除停用词
knn_tfid_stop = KNeighborsClassifier()
knn_tfid_stop.fit(x_tfid_stop_train, y_train)   # 学习
knn_tfid_stop_y_predict = knn_tfid_stop.predict(x_tfid_stop_test)  # 预测

# 5 模型评估
print("--------------KNN的结果--------------")
# 对普通统计CountVectorizer提取的特征学习模型进行评估
print("未去除停用词的CountVectorizer提取的特征学习模型准确率：", knn_count.score(x_count_test, y_test))
print("更加详细的评估指标:\n", classification_report(knn_count_y_predict, y_test))
print("去除停用词的CountVectorizer提取的特征学习模型准确率：", knn_count_stop.score(x_count_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(knn_count_stop_y_predict, y_test))

# 对TfidVectorizer提取的特征学习模型进行评估
print("TfidVectorizer提取的特征学习模型准确率：", knn_tfid.score(x_tfid_test, y_test))
print("更加详细的评估指标:\n", classification_report(knn_tfid_y_predict, y_test))
print("去除停用词的TfidVectorizer提取的特征学习模型准确率：", knn_tfid_stop.score(x_tfid_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(knn_tfid_stop_y_predict, y_test))



