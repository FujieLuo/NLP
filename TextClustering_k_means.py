# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:43:09 2020

@author: 罗福杰
"""
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

#根据数据地址的路径，获取路径下 文件夹的名字列表
def get_folder_list(root_path):
    folder_list = []
    root_path =root_path
    #读取该目录下所有文件夹的名字，并组成一个列表
    folder_list = os.listdir(root_path)
    return folder_list

#根据文件夹的列表，返回包含所有文件名的列表
def get_files_name_list(root_path):
    
    #得到了 20 个类别的标签
    class_20_names = get_folder_list(root_path)
    #print(class_20_names)

    files_list = []
    
    for per_class_folder in class_20_names:
        
        one_class_path_list = []

        #一个文件夹的路径
        one_class_path = str(root_path+"/"+per_class_folder+'/')
        #一个文件夹下的文件列表 
        one_class_list = get_folder_list(one_class_path)
        #print(one_class_list)
        
        for one_text in one_class_list:
            one_text_path = str(one_class_path+one_text)
            one_class_path_list.append(one_text_path)
            #print(one_text_path)
        
        files_list.append(one_class_path_list)
    #print(files_list)
    #返回两个参数，都是list的形式，
    # class_20_names是20个类别的名称，files_list 是每个类别的文件名[[],[],[]]：
    return class_20_names,files_list

'''读一个文件的内容'''
def read_one_text_file(one_text_path):
    with open(one_text_path, 'r',encoding='utf-8', errors='ignore') as f:
        one_text_info = f.read()	
    return one_text_info

'''获取所有news的信息，并放在一个字典news里面'''
def get_news_info(root_path):
    
    class_20_names,files_list = get_files_name_list(root_path)  
    news = {"data_path":[],
            "data":[],
            "target_names":[],
            "target":[],
            'data_clean_words':[],
            'data_tf_idf':[],
            'data_no_repeat':[],
            'data_tf_idf_10':[],
            'data_no_repeat':[]
            }
    
    #对于每个类别的每一个文件,添加到news字典里
    for i in range(0,len(class_20_names)):
        for j in range(0,len(files_list[i])):
            news["data_path"].append(files_list[i][j])
            news["data"].append(read_one_text_file(files_list[i][j]))
            news["target_names"].append(class_20_names[i])
            news["target"].append(i)
            
    return news

'''去除停用词'''
def remove_stop_words(words_list,stop_words_path):
    stop_words_list = []
    words_clean_list = []
    f = open(stop_words_path,'r')
    lines = f.readlines()
    for line in lines:
        stop_words_list.append(line.strip('\n'))
    
    for word in words_list:
        if word in stop_words_list:
            continue
        words_clean_list.append(word)
    
    return words_clean_list

'''构建语料库，是一个list'''
def make_words_dataset(words_two_list):
    words_dataset_list = []

    for i in range(0,len(words_two_list)):
        print("一共有  ",len(words_two_list)," 个文档，正在处理第：  ", i+1, " 个文档")
        for word in words_two_list[i]:

            if word not in words_dataset_list:
                words_dataset_list.append(word)
    
    print("****************Done!***************")
    print("语料库中的单词总数是： ",len(words_dataset_list))
    print("语料库中前100个单词：\n",words_dataset_list[:100])
    return words_dataset_list

'''包含该词的文档数'''
def num_include_this_word(words_two_list,word):
    num_text_include_this_word = 0
    for i in range(0,len(words_two_list)): #对于每篇文章
        if word in words_two_list[i]:
            num_text_include_this_word += 1
            
    return num_text_include_this_word


'''计算TF-IDF'''
def compute_tf_idf(words_two_list,path,N):
    
    all_text_tfidf_list = []
    all_text_no_repeat_list =[]
    all_text_tfidf_list_10 = []
    all_text_no_repeat_list_10 =[]
    text_no_repeat_word_less_N = []
    
    
    
    
    for i in range(0,len(words_two_list)):#对每一篇文章
        #print("******************************")
        #if i == 400:
           #break
        print("一共有  ",len(words_two_list)," 个文档，正在计算第：  ", i+1, " 个文档的 TF-IDF ")
        
        
        this_text_word_num = len(words_two_list[i])
        
        #将这篇文章里不重复的元素挑选出来
        this_text_word_no_repeat = list(set(words_two_list[i]))
        this_text_tfidf_numpy = np.zeros([3,len(this_text_word_no_repeat)])
        
        #将TF-IDF为前十的元素取出来
        this_text_tfidf_numpy_10 =np.zeros(N)
        this_text_word_no_repeat_10 = []
        
        #print("总词数，不重复的词数",this_text_word_num,len(this_text_word_no_repeat))
        
        for j in range(0,len(this_text_word_no_repeat)):  #对于每一个词
            word = this_text_word_no_repeat[j]    
            #计算TF
            #print("当前的词：",word)
            this_word_num = words_two_list[i].count(word)
            #print("这个词在这篇文章中出现的次数：",this_word_num)
            tf_this_word = this_word_num / this_text_word_num
            this_text_tfidf_numpy[1][j] = tf_this_word
            
            #计算IDF
            num_text_include_this_word = num_include_this_word(words_two_list,word)
            #print("这个词在所有文章中出现的次数：",num_text_include_this_word)
            idf_this_word = math.log(len(words_two_list),(num_text_include_this_word+1))
            this_text_tfidf_numpy[2][j] = idf_this_word
            
            this_text_tfidf_numpy[0][j] = tf_this_word * idf_this_word
            
        
        if len(this_text_word_no_repeat) < N:
            for u in range(0,len(this_text_word_no_repeat)):
                this_text_tfidf_numpy_10[u] = this_text_tfidf_numpy[0][u]
                this_text_word_no_repeat_10 = this_text_word_no_repeat
                text_no_repeat_word_less_N.append(path[i])
        else:
            #将TF-IDF为前十的元素取出来,作为特征
            idx = np.argpartition(this_text_tfidf_numpy[0], -N)[-N:]
            this_text_tfidf_numpy_10 = this_text_tfidf_numpy[0][idx]
            #print(type(this_text_tfidf_numpy_10))   #<class 'numpy.ndarray'>
            for p in idx:
                this_text_word_no_repeat_10.append(this_text_word_no_repeat[p])
        
        all_text_tfidf_list_10.append(this_text_tfidf_numpy_10)
        all_text_no_repeat_list_10.append(this_text_word_no_repeat_10)
        
        all_text_tfidf_list.append(this_text_tfidf_numpy)
        all_text_no_repeat_list.append(this_text_word_no_repeat)
            
    return all_text_tfidf_list,\
        all_text_no_repeat_list,\
        all_text_tfidf_list_10,\
        all_text_no_repeat_list_10,\
        text_no_repeat_word_less_N

'''对一个文本进行一系列操作：去掉无用的符号，切分，去除停用词
最终返回的是words_clean_list，这个文本去掉停用词后的单词列表'''
def spilt_words(text,stop_words_path): 

    # 将源文本进行切分，遇到第一个含两个分行的切分一次，去掉头信息
    text_list = text.split('\n\n',1)
    text = text_list[-1]    
    text = text.replace("\n\n",'').replace("\n",'').replace("|>", "").replace('.', '')
    r='[’!"#$%&\'()*+_,-./:;<=>?@[\\]^`{|}~\d]+'
    text = re.sub(r,' ',text)
    text = text.replace("   ",' ').replace("  ",' ')
    text = text.lower()  #转为小写字符
    words_list = text.split(' ')    
    words_list_new =[]
    for word in words_list:    #去掉分割出来为空的字符，如：''这种字符
        if word =="":
            continue
        words_list_new.append(word)
    words_list = words_list_new
    
    words_clean_list = remove_stop_words(words_list, stop_words_path)

    return words_clean_list

#计算欧氏距离
def distance(e1,e2):

    return np.linalg.norm(e1 - e2)

# arr中距离a最远的元素，用于初始化聚类中心
def farthest(k_arr, arr,N,arr_path):
    f = np.zeros([1,N])
    max_d = 0
    for v in range(0,len(arr)):
        e = arr[v]
    #for e in arr:
        d = 0
        for i in range(k_arr.__len__()):
            d = d + distance(k_arr[i],e)
        if d > max_d:
            max_d = d
            f = e
            f_path = arr_path[v]
    return f,f_path

# 集合中心
def means(arr,N):
    num_bit = N
    num_mean_list = [] 
    for i in range(num_bit):
        num_mean_list.append(np.mean([e[i] for e in arr]))
    num = np.array(num_mean_list)
    return num

'''#开始聚类，k是类别的个数，x_list是数据向量,
   #n是迭代次数,N是每个文章的维度参数'''
   #cluster_k_means(20,news['data_tf_idf_10'],n=10,N):
def cluster_k_means(k,x_list,n,N,path):
    point_num = len(x_list)
    arr = np.array(x_list)
    arr_path = path
    #初始化聚类中心和聚类容器
    r= np.random.randint(point_num-1)
    k_arr = np.array([arr[r]]) # 随机选一个当作聚类中心
    k_arr_path = [arr_path[r]]   #对应的路径的中心
    
    cla_arr = [[]]
    cla_arr_path = [[]]
    cla_info = {'cla_arr':cla_arr,'cla_arr_path':cla_arr_path}
    
    for i in range(k-1):
        d,d_path = farthest(k_arr,arr,N,arr_path)
        k_arr = np.concatenate([k_arr,np.array([d])])
        k_arr_path.append(d_path)
        cla_arr.append([])
        cla_arr_path.append([])
    #print("初始化的聚类中心：",k_arr)
    
    #迭代聚类
    cla_temp = cla_arr
    cla_temp_path = cla_arr_path
    
    print(k_arr_path)
    print(k_arr)
    print(arr)
    
    for i in range(n):  # 迭代n次
        print("一共有 ",n," 次迭代，现在正进行第 ",i+1 ," 轮迭代...")
        for v in range(0,len(arr)):
            e = arr[v]
        #for e in arr :  # 把集合里每一个元素聚到最近的类
            ki = 0  # 假定距离第一个中心最近
            min_d = distance(e, k_arr[ki])
            #print(min_d)
            for j in range(1,k_arr.__len__()):
                if distance(e, k_arr[j]) < min_d:
                    min_d = distance(e, k_arr[j])
                    ki = j
            cla_temp[ki].append(e)
            cla_temp_path[ki].append(arr_path[v])
            
            #print(cla_temp[ki])
            #print("")
        
        # 迭代更新聚类中心
        for kk in range(k_arr.__len__()):
            if n-1 == i:
                break
            
            k_arr[kk] = means(cla_temp[kk],N)
            cla_temp[kk] = []
            cla_temp_path[kk] = []
    
    # 输出结果 
    print("***************************")
    print(k)
    for ii in range(0,k): #对于每个类别
        print("这是第 ",ii+1," 类，这个类的真值是：")
        print(cla_temp_path[ii])
        print("")
        
    
    if  N==2:
        col = ['HotPink', 'Aqua', 'Chartreuse', 'yellow', 'LightSalmon',
               'Crimson','Purple','Indigo','DarkSlateBlue','Blue',
               'GadetBlue','DarkCyan','SeaGreen','OliveDrab','Olive',
               'Gold','Orange','SaddleBrown','Black','Gray']
        for i in range(k):
            #print(k_arr[i][0])
            #print("k_arr[i][1]",k_arr[i][1])
            plt.scatter(k_arr[i][0], k_arr[i][1], linewidth=10, color=col[i])
            plt.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], color=col[i])
        plt.show()
    return
    

def main():
    
    root_path = 'C:/Users/admin/Desktop/mini_newsgroups'
    stop_words_path = 'C:/Users/admin/Desktop/english_stop_words/stopwords_2.txt'
    news = get_news_info(root_path)
    N = 20 #每篇文章的TF-IDF维度

    words_clean_list = spilt_words(news["data"][10],stop_words_path)
    
    for i in range(0,len(news['data'])):#对于每一篇文章
        words_clean_list = spilt_words(news["data"][i],stop_words_path)
        news['data_clean_words'].append(words_clean_list)
        
    #print(news['data'][5])
    #print(news['data_clean_words'][5])
    
    #计算语料库
    #words_dataset_list = make_words_dataset(news['data_clean_words'])
    #计算TF-IDF
    ret = compute_tf_idf(news['data_clean_words'],news['data_path'],N)
    
    all_text_tfidf_list,\
    all_text_no_repeat_list,\
    all_text_tfidf_list_10,\
    all_text_no_repeat_list_10,\
    text_no_repeat_word_less_N=ret
    
    news['data_tf_idf'] = all_text_tfidf_list
    news['data_no_repeat'] = all_text_no_repeat_list
    news['data_tf_idf_10'] = all_text_tfidf_list_10
    news['data_no_repeat_10'] = all_text_no_repeat_list_10
    
    #print(news['data_tf_idf_10'])
    #print(news['data_no_repeat_10'])
    
    #开始进行聚类
    #arr = np.random.randint(100, size=(100, 1, 2))[:, 0, :]
    #cluster_k_means(k=4,x_list=arr,n=10,N=N,path=news['target'][:500])

    cluster_k_means(k=20,x_list=news['data_tf_idf_10'],n=10,N=N,path=news['target'][:2000])
    '''
    km_model = KMeans(n_clusters=3)    # 实例化构造聚类器
    km_model.fit(news['data_tf_idf_10'])                  # 聚类，传入所有特征数据
    label_pred = km_model.labels_   #获取聚类的标签
    print("--------------------------------")
    print(label_pred)
    print("len:",len(label_pred))
    print("第十个预测的结果:",label_pred[10])
    '''
    print("不满足要求的：",len(text_no_repeat_word_less_N))
    #for e in text_no_repeat_word_less_N:
        #print(e)
        
if __name__ == '__main__':
    main()


