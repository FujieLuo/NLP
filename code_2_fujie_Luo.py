# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:08:54 2020

@author: admin
"""


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
from sklearn.metrics import homogeneity_score,adjusted_rand_score,adjusted_mutual_info_score
from sklearn.metrics import completeness_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans



# 1 下载新闻数据
news = fetch_20newsgroups(subset="all")
'''
print("news:",news.keys())
print("data1:",news.data[1])
print("data1-filename:",news.filenames[1])
print("target_names:",news.target_names[1])
print("target:",news.target[1])
print("DESCR:",news.DESCR[1])
print("type:",type(news))
print("")
'''


# 2 分割训练数据和测试数据
x_text, x_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.20,
                                                    random_state=30)

print("----------------------")
print(type(x_text))
print(len(x_text))
print(x_text[1])
print("")

'''
print("----------------------")
print(type(y_train))
print(len(y_train))
print(y_train[1])
print("")
'''


# 去除停用词
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
x_tfid_stop_train = tfid_stop_vec.fit_transform(x_text)

print("**********************")
print(type(x_tfid_stop_train))
print(x_tfid_stop_train.shape)
print(x_tfid_stop_train)
print(type(x_tfid_stop_train[1]))
print("")



km_model = KMeans(n_clusters=20)    # 实例化构造聚类器
km_model.fit(x_tfid_stop_train)                  # 聚类，传入所有特征数据
label_pred = km_model.labels_   #获取聚类的标签
print("--------------------------------")
print(label_pred)
print("len:",len(label_pred))
print("前1000样本聚类的结果:",label_pred[:1000])

print("评价指标：兰德指数")
print(adjusted_rand_score(y_train, label_pred))
print("")


print("评价指标：互信息")
print(adjusted_mutual_info_score(y_train, label_pred))
print("")



print("评价指标：同质性homogeneity，每个群集只包含单个类的成员。")
print(homogeneity_score(y_train, label_pred))
print("")




print("评价指标：完整性completeness：给定类的所有成员都分配给同一个群集")
print(completeness_score(y_train, label_pred))
print("")

