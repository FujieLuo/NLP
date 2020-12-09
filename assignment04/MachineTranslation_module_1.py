# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:59:17 2020

@author: 罗福杰
"""
import math
import re
import csv


def create_cn_en_words(corpus):

    # 设置英文外文词汇表
    english_vocab = []
    foreign_vocab = []
    for sp in corpus:
        for fw in sp[0]:
            foreign_vocab.append(fw)
        for ew in sp[1]:
            english_vocab.append(ew)
    english_words = sorted(list(set(english_vocab)), key=lambda s:s.lower())
    foreign_words = sorted(list(set(foreign_vocab)), key=lambda s:s.lower())
    
    return english_words,foreign_words


def create_words_list(path_en,path_cn):
    corpus = []
    list_en_words = []
    list_cn_words = []
    
    # 读取中英文文件，处理后将每个词并存入list中。
    with open(path_en,'r',encoding='utf-8',errors='ignore') as f:
        for line in f.readlines():
            words_list_new1 = []
            line = line.replace('\n','')
            r='[’!"¡#$%&\'()*+_,-./:;<=>?@[\\]^`{|}~\d]+'
            line = re.sub(r,' ',line)
            line = line.replace('ª',' ').replace('ó÷', '')
            line = line.lower()  #转为小写字符
            words_list = line.split(' ') 
            for word in words_list:    #去掉分割出来为空的字符，如：''这种字符
                if word =="":
                    continue
                words_list_new1.append(word)
            #print(words_list_new)
            list_en_words.append(words_list_new1)
    
    with open(path_cn,'r',encoding='utf-8',errors='ignore') as f:
        for line in f.readlines():
            words_list_new2 =[]
            line = line.strip('\n')
            r="[^0-9A-Za-z\u4e00-\u9fa5]"
            line = re.sub(r,' ',line)
            words_list = line.split(' ') 
            for word in words_list:    #去掉分割出来为空的字符，如：''这种字符
                if word =="":
                    continue
                words_list_new2.append(word)
            #print(words_list_new)
            list_cn_words.append(words_list_new2)
    
    # 构建语料库
    #for i in range(0,5):
    for i in range(0,len(list_cn_words)):
        sentense_cn_and_en = []
        sentense_cn_and_en.append(list_cn_words[i])
        sentense_cn_and_en.append(list_en_words[i])
        corpus.append(sentense_cn_and_en)
    
    return(corpus)


# 给定e,f句子和t,计算p(e|f)
def probability_e_f(e, f, t, epsilon=1):
    l_e = len(e)
    l_f = len(f)
    p_e_f = 1
    for ew in e:
        t_ej_f = 0
        for fw in f:
            t_ej_f += t[fw][ew]
        p_e_f = t_ej_f * p_e_f
        
    p_e_f = p_e_f * epsilon / ((l_f+1)**l_e)
    return p_e_f

# 输入语料库计算perplexity
def perplexity(corpus, t, epsilon=1):
    log2pp = 0
    for sp in corpus:
        prob = probability_e_f(sp[1], sp[0], t)
        log2pp += math.log(prob, 2) 
    
    pp = 2.0 **(-log2pp)
    return pp

def create_t(corpus,init_val):
    t = {}
    for sp in corpus:
        for f_w in sp[0]:
            for e_w in sp[1]:
                if f_w not in t:
                    t[f_w] = {}
                t[f_w][e_w] = init_val
    
    return t

def main():
    
    path_en = 'C:/Users/admin/Desktop/english_stop_words/en.txt'
    path_cn = 'C:/Users/admin/Desktop/english_stop_words/cn.txt'
    
    corpus = create_words_list(path_en,path_cn) 
    #print("corpus[0:100]:\n",corpus[0:100])
    #corpus = [[['我的','书'],['my','book']],[['一本', '杂志'],['a', 'magazine']],[['我的', '杂志'],['my', 'magazine']]]
    #corpus = [[['一本','书'],['a','book']],[['一本', '杂志'],['a', 'magazine']]]
    #corpus = [[['这是','我的','书'],['this','is','my','book']],[['一本', '杂志'],['a', 'magazine']]]
    english_words,foreign_words = create_cn_en_words(corpus)
    
    init_val = 0.7
    # 用来保存不同外文单词翻译成不同英文单词的概率
    t = create_t(corpus,init_val)
    #print(t)
    # print("foreign的总个数有：",len(t))
    # n = 0
    # for i, (k, v) in enumerate(t.items()):
    #     for j in range(0,len(t[k])):
    #         n=n+1
    # print("序列对的总个数有：",n)
      
    num_epochs = 15
    s_total = {}
    
    result_info = []
    
    #perplexities = []
    for epoch in range(num_epochs):
        print("--------epoch % s--------" % (epoch + 1))
        #perplexities.append(perplexity(corpus, t))
        count = {}
        total = {}
        
        #for fw2 in foreign_words:
        #    total[fw2] = 0.0
        
        for sp in corpus:
            for f_w in sp[0]:
                total[f_w] = 0.0
                for e_w in sp[1]:
                    if f_w not in count:
                        count[f_w] = {}
                    count[f_w][e_w] = 0.0
                    
        # n2=0
        # for i, (k, v) in enumerate(count.items()):
        #     for j in range(0,len(count[k])):
        #         n2=n2+1
        # print("序列对的总个数有：",n2)
        for i in range(0,len(corpus)):
            sp1 = corpus[i]
            if i % 1000 == 0:
                print("epoch : ", epoch,"ALL: ",len(corpus),"   NOW processing : ",i)
        #for sp1 in corpus:
            # s_total[a]相当于上面手工推演中的t(a|一本)+t(a|书)  
            # s_total也就相当于c(e)
            
            for ew1 in sp1[1]:
                #print(ew1)
                s_total[ew1] = 0.0
                for fw1 in sp1[0]:
                    s_total[ew1] += t[fw1][ew1] 
           
            # 此时计算出的count[一本][a]相当于c(a|一本)，在不同语料中继续相加相同对应的概率  
            # total[一本]相当于一本翻译成不同英文词的概率和
            # 还要考虑不同语料，主要用于t概率矩阵的归一化
            for ew2 in sp1[1]:
                for fw2 in sp1[0]:
                    count[fw2][ew2] += t[fw2][ew2] / s_total[ew2]
                    total[fw2] += t[fw2][ew2] / s_total[ew2]
                
        # 概率的归一化，使得一个外文单词翻译成不同英文单词的概率和为1
        for sp2 in corpus:
            for fw in sp2[0]:
                for ew in sp2[1]:
                    t[fw][ew] = count[fw][ew] / total[fw]

                
        for fw in t:
            #print('foreign word: ', fw)
            sorted_list = sorted(t[fw].items(), key=lambda x:x[1], reverse=True)
            #print(fw," ",sorted_list[0][0],sorted_list[0][1])
            #for (ew, p) in sorted_list:
                #print('prob to %s \tis %f'%(ew, p))
            #print('')
        if epoch == num_epochs-1:
            
            for fw in t:
                one_info = []
                sorted_list = sorted(t[fw].items(), key=lambda x:x[1], reverse=True)
                one_info.append(fw)
                one_info.append(sorted_list[0][0])
                one_info.append(sorted_list[0][1])
                result_info.append(one_info)
            
    #plt.plot(perplexities) # 需要matplotlib库
    # 将预测的结果从大到小进行排序
    result_sorted_list = sorted(result_info, key=lambda r:r[2], reverse=True)
    #print(result_sorted_list)
    
    f = open('result_info_all_epoch_15.csv','w',encoding='utf-8-sig',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["汉语","英语","概率"])
    for e in result_sorted_list:
        csv_writer.writerow(e)
    f.close()
    
    
if __name__ == '__main__':
    main()

