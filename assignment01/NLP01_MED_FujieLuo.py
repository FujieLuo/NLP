#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:29:38 2020

@author: fujie
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from tkinter import messagebox


window = tk.Tk()
window.title("NLP作业①   最小编辑距离计算器 by 罗福杰")
#设置窗口的长和宽不可调节
window.resizable(0,0)
# 设置初始界面的长和高
window.geometry('700x600')


#lb_title = tk.Label(window,text='Welcome to "最 小 编 辑 距 离 计 算 器" hhaha',bg='green',font =('Arial',16),width=80,height=2)
lb_title = tk.Label(window,text='Welcome to "Minimum Edit Distance Calculator"',bg='green',font =('Time New Roman',16),width=80,height=2)
#lb_title = tk.Label(window,text='欢迎使用最小编辑距离计算器',font =('Arial',18),width=40,height=20)
lb_title.pack(side='top')
#lb_title.pack(side='bottom')

lb_mode = tk.Label(window,text = 'Hello ! Please enter two words below ( length<=255 )',font =('Time New Roman',17))
#lb_mode.pack(side='left')
#设置标签的位置
lb_mode.place(x=10,y=80,anchor='nw')


#cmb = ttk.Combobox(window)
#cmb.place(x = 210,y = 85,anchor='nw')
#cmb['value']=('English','中文')
#设置默认值
#cmb.current(0)


lb_char1=tk.Label(window,text='Word 1 :',font=('Time New Roman',14))
lb_char1.place(x=20,y=180)
lb_char2=tk.Label(window,text='Word 2 :',font=('Time New Roman',14))
lb_char2.place(x=20,y=295)


lb_result = tk.Label(window,text='Result of The Minimum Edit Distance : ',font=('Time New Roman',16))
lb_result.place(x=10,y=500)

result_value = tk.StringVar()
lb_result_value = tk.Label(window,textvariable=result_value,font=('Time New Roman',30))
lb_result_value.place(x=450,y=490)

result_lb = tk.StringVar()
lb_result_lb =tk.Label(window,textvariable=result_lb,font=('Time New Roman',20))
lb_result_lb.place(x=500,y=100)

result_process_lb=tk.StringVar()
lb_result_process=tk.Label(window,textvariable=result_process_lb,font=('Time New Roman',20))
lb_result_process.place(x=350,y=150)

#entry_word1=tk.Entry(window,bd=3,xscrollcommand=True)
#entry_word1.place(x=150,y=185)
#entry_word2=tk.Entry(window,bd=3)
#entry_word2.place(x=150,y=245)

f=tk.Frame(window)
s1 = tk.Scrollbar(f,orient=tk.VERTICAL)  
tx_word1=tk.Text(window,autoseparators=2,font =('Time New Roman',16),height=3,width=20,relief='groove',fg='red')
tx_word1.place(x=150,y=185)


tx_word2=tk.Text(window,autoseparators=2,font =('Time New Roman',16),height=3,width=20,relief='groove',fg='red')
tx_word2.place(x=150,y=300)





def minDistance(w1,w2):
    w1=w1.strip()
    w2=w2.strip()
    m,n = len(w1),len(w2)
    if (w1=="" or m==0):
        messagebox.showinfo('Warning!!!','请在Word1中输入字符！')
        return m
    if m>255:  
        messagebox.showinfo('Warning!!!','字符长度不得超过255！')
        return m
    if (w2=="" or n==0):
        messagebox.showinfo('Warning!!!','请在Word2中输入字符！')
        return n
    if n>255:  
        messagebox.showinfo('Warning!!!','字符长度不得超过255！')
        return n
    
    # 生成全零矩阵，形状是（m+1, n+1）
    #step = [[0]*(n+1) for _ in range(m+1)]
    step = np.zeros([m+1,n+1])
    
    for i in range(1,m+1):
        step[i][0] = i
    
    for j in range(1,n+1):
        step[0][j] = j
    
    for i in range(1,m+1):
        
        for j in range(1,n+1):
            if w1[i-1] == w2[j-1]:
                diff = 0
            else:
                diff = 1
            step[i][j] = min(step[i-1][j-1],min(step[i-1][j],step[i][j-1]))+diff
    
    #return step[m][n]
    return step,int(step[m][n])


def backtrackingPath(word1,word2):
    dp,mindista = minDistance(word1,word2)
    m = len(dp)-1
    n = len(dp[0])-1
    operation = []
    spokenstr = []
    writtenstr = []
    
    operation_process = []
    
    back_way = np.zeros([m+1,n+1])
    back_way[m][n] = 1 

    while n>=0 or m>=0:
        
        if n and dp[m][n-1]+1 == dp[m][n]:
            processer="Insert : \""+(word2[n-1])+'\".'
            operation_process.append(processer)
            spokenstr.append("Insert")
            writtenstr.append(word2[n-1])
            operation.append("NULLREF:"+word2[n-1])
            n -= 1
            back_way[m][n] = 1 
            continue
        
        if m and dp[m-1][n]+1 == dp[m][n]:
            processer="Delete : \""+(word1[m-1])+'\".'
            operation_process.append(processer)
            spokenstr.append(word1[m-1])
            writtenstr.append("Delete")
            operation.append(word1[m-1]+":NULLHYP")
            m -= 1
            back_way[m][n] = 1 
            continue
        
        if dp[m-1][n-1]+1 == dp[m][n]:
            processer="Replace : \""+(word1[m-1])+'\" To \"'+(word2[n-1])+'\".'
            operation_process.append(processer)
            spokenstr.append(word1[m - 1])
            writtenstr.append(word2[n-1])
            operation.append(word1[m - 1] + ":"+word2[n-1])
            n -= 1
            m -= 1
            back_way[m][n] = 1 
            continue
        if dp[m-1][n-1] == dp[m][n]:
            spokenstr.append(' ')
            writtenstr.append(' ')
            operation.append(word1[m-1])
        n -= 1
        m -= 1
        back_way[m][n] = 1 
        
    spokenstr = spokenstr[::-1]
    writtenstr = writtenstr[::-1]
    operation = operation[::-1]
    # print(spokenstr,writtenstr)
    # print(operation)
    return spokenstr,writtenstr,operation,operation_process,back_way

def compute():
    
    #word1=entry_word1.get()
    #word2=entry_word2.get()
    word1=str(tx_word1.get(1.0, "end"))
    word2=str(tx_word2.get(1.0, "end"))

    step,mindis=minDistance(word1,word2)
    result_value.set(mindis)
    
    btn_show.place(x=200,y=400)
    

def show_operation():
    compute()
    word1=tx_word1.get(1.0, "end")
    word2=tx_word2.get(1.0, "end")
    
    window_show=tk.Toplevel(window)
    window_show.geometry('800x600')
    window_show.title('最小编辑距离实现的具体过程')
    
    lb_show_Process=tk.Label(window_show,text='Process : ',font =('Time New Roman',18))
    lb_show_Process.place(x=30,y=40)
    
    tx_show=tk.Text(window_show,autoseparators=2,font =('Time New Roman',16),height=20,width=25)
    tx_show.place(x=25,y=80)
    
    spokenstr,writtenstr,operation,operation_process,back_way=backtrackingPath(word1,word2)
    #result_process_lb.set(operation_process)
    
    tx_show.insert(1.0," ")
    process=str(operation_process)
    process=process.replace('.\',', '\n')
    process=process.replace('.\']', "")
    process=process.replace('\'', "")
    process=process.replace('[', "")
    tx_show.insert(1.1,process)
    
    stepff,_=minDistance(word1, word2)
    lines,column_num=stepff.shape[0],stepff.shape[1]
            
    columnskk=[str(int(stepff[0][i])) for i in range(0,column_num)]
    tree = ttk.Treeview(window_show, show = "headings", columns = columnskk,height=lines, selectmode = tk.BROWSE)
    tree["columns"]=columnskk
    #设置每一列的属性
    for i in range(0,column_num):
        tree.column(columnskk[i], width=25)
    # 开始填充数据
    for i in range(0,lines):
        step_line_val=[str(int(stepff[i][j])) for j in range(0,column_num)]
        tree.insert('','end',values=step_line_val)

    tree.place(x=300,y=80)
    

    back_way_only_1 = np.zeros([lines,column_num])
    for i in range(0,lines):
        for j in range(0,column_num):
            back_way_only_1[i][j]=stepff[i][j]*back_way[i][j]
    
    back_way_only_1_str = [['0']*column_num for _ in range(lines)]
    for i in range(0,lines):
        for j in range(0,column_num):
            if back_way[i][j]==1:
                back_way_only_1_str[i][j]=str(int(stepff[i][j]))
            else:
                back_way_only_1_str[i][j]=' '
    back_way_only_1_str[0][0]='0'
    
    tree2 = ttk.Treeview(window_show, show = "headings", columns = columnskk,height=lines, selectmode = tk.BROWSE)
    tree2["columns"]=columnskk
    #设置每一列的属性
    for i in range(0,column_num):
        tree2.column(columnskk[i], width=25,anchor='e')
    
    for i in range(0,lines):
        tree2.insert('','end',values=back_way_only_1_str[i])
    
    tree2.place(x=600,y=80)
    

btn_compute = tk.Button(window,text ='Start Compute',width=15,height=5,command=compute)
btn_compute.place(x=30,y=400)
btn_show = tk.Button(window,text ='Show Operation',width=15,height=5,command=show_operation)
#btn_show.place(x=200,y=400)
btn_show.place_forget()

window.mainloop()