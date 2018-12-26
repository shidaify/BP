#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import sys
import math
import time
import struct
import pygame
import numpy as np
from PIL import Image
import pylab as pl
from tqdm import *

mode = True 
y_target=np.identity(10)
drawing = False  # 是否开始画图
start = (-1, -1)
size=700
# In[2]:


def load_mnist(path, kind='train'):
    """读取mnist"""
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"% kind)
    
    images_path = os.path.join(path,"%s-images.idx3-ubyte"% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    
   # print(images[labels==5][0].reshape(1,784))
    return images, labels


# In[3]:


def read_fzy():
    image = []
    label = []
    for cou in range(1,21):
        for num in range(10):
            img_name = 'fzy/' + str(num) + '-' + str(cou) + '.png'
            img = cv2.imread(img_name,0)
            tmp = 255-np.array(img)
            tmp = np.reshape(tmp,(1,784)).tolist()[0]
            image.append(tmp)
            label.append(num)
    return image,label


# In[4]:


def read_lc():
    image = []
    label = []
    for i in range(1,362):
        img_name = 'lc/handled/' + str(i) + '.png'
        img = cv2.imread(img_name,0)
        tmp = 255-np.array(img)
        tmp = np.reshape(tmp,(1,784)).tolist()[0]
        image.append(tmp)
    with open('lc/answers.txt','rb') as f:
        for each in f:
            each = int(each[0:1])
            label.append(each)
    return image,label


# In[5]:


def read_xy():
    image = []
    label = []
    for cou in range(1,501):
        for num in range(10):
            img_name = 'xy/' + str(num) + '/' + str(num) + ' (' + str(cou) + ')' + '.png'
            img = cv2.imread(img_name,0)
            img = cv2.resize(img,(28,28),3)
            tmp = np.reshape(img,(1,784)).tolist()[0]
            image.append(tmp)
            label.append(num)
    return image,label


# In[6]:


def deal(x,w,m,n,b):#x输入，w权值
    """处理上一层到下一层,得到下一层输入"""
    #h1=[]
    hi=np.dot(np.mat(w),np.mat(x).T)
    hi=hi.T+np.mat(b)
    h=hi.tolist()[0]
    return  h


# In[7]:


def sigmoid(hi):
    """输入本层输入值，返回其输出值"""
    ho=[]
    for i in range(0,len(hi)):
        num=1/(1+np.exp(-hi[i]))
        ho.append(num)
    return ho


# In[8]:


def eta_fun(k):
    eta_new=eta*(1/(1+k/100))
    return eta_new


# In[9]:


def back2(w,label,h,y,b2):#w为从隐含层到输出层的权值，target为输出层目标输出值，h为隐含层输出,y为输出层实际输出
    """计算从输出层到隐含层"""
    global h_num,y_num,eta,y_target
#     w_n=[]
#     w_m=[]
   
    w_dif=np.mat(np.multiply((np.array(y)-np.array(y_target[label])),np.multiply(np.array(y),(1-np.array(y)))))
    
    b2=b2-eta*w_dif
    
    w_dif=w_dif.T
    w_diff=np.dot(w_dif,np.mat(h))
    w_new=np.array(w)-eta*(np.array(w_diff))
   # for m in range(0,y_num):
    #    for n in range(0,h_num):
           
      #      w_dif=-(y_target[label][m]-y[m])*y[m]*(1-y[m])*h[n]
          #  w_new=w[m][n]-eta*w_dif
        #    w_n.append(w_new)
      #  w_m.append(w_n)
    return w_new,b2


# In[10]:


def back1(w_1,w_2,label,x,h,y,b1):
    """计算从隐含层到输入层"""
    global h_num,x_num,y_target,y_num
    
#     E_ho=np.dot((np.multiply((np.mat(y)-np.mat(y_target[label])),np.multiply(np.mat(y),(1-np.mat(y))))),np.mat(w_2))
#     E_ho=E_ho.tolist()[0]

    E_ho=[]

    for n in range(0,h_num): 
        E_sum=0
        for m in range(0,y_num):
            E=-(y_target[label][m]-y[m])*y[m]*(1-y[m])*w_2[m][n]
            E_sum=E+E_sum
        E_ho.append(E_sum)

    
    w_dif=np.mat(np.multiply(np.array(E_ho),np.multiply(np.array(h),(1-np.array(h)))))
    
    b1=b1-eta*w_dif
    
    w_dif=w_dif.T
    w_diff=np.dot(w_dif,np.mat(x))
    w_new=np.array(w_1)-eta*(np.array(w_diff))

   # for a in range(0,h_num):
     #  for b in range(0,x_num):
    #        w_dif=E_ho[a]*ho[a]*(1-ho[a])*x[b]
    #        w_new=w_1[a][b]-eta*w_dif
    #        w_b.append(w_new)
    #    w_a.append(w_b)
    return w_new,b1


# In[11]:


def lost_target(label,yo):
    """损失函数计算"""
    global y_target
    yo=np.array(yo)
    y_t=y_target[label]
    sum=(yo-y_t)
    e_array=sum*sum
    E=0.5*e_array.sum(axis=0)
    
    ln_a=[]
    for i in yo:
        ln_a.append(math.log(i))
    ln_1a=[]
    for i in (1-yo):
        ln_1a.append(math.log(i))
    ln_a=np.array(ln_a)
    ln_1a=np.array(ln_1a)
    a=y_t*ln_a+(1-y_t)*(ln_1a)
    C=-0.1*(a.sum(axis=0))
    return E,C


# In[12]:


global h_num,x_num,y_num,eta
h_num = 90
x_num = 784
y_num = 10
eta = 0.22

paint_E=[]
paint_C=[]
b1=np.zeros(h_num)
b2=np.zeros(y_num)

w1=0.01*np.random.random((h_num,x_num))-0.005#w1为输入层到隐含层的各个权值，h_num为隐含层所有的节点数量，wn为第m个节点中的所有权值
w2=0.01*np.random.random((y_num,h_num))-0.005#w2为隐含层到输出层的各个权值，y_num为输出层所有节点，h_num为第m个节点中所有的权值

# w1=np.loadtxt('w1.txt')
# w2=np.loadtxt('w2.txt')
# flag=1

#get=input("是否读取数据：\n1：读取新数据\n2：用原数据实验\n")
get='2'#权值阈值已完成存储，暂不需要重新运行
if(get=='1'):
    #x_train,y_train=load_mnist("C:\\Users\\surface\\Desktop\\mnist")
    x_train,y_train=load_mnist("C:\\Users\\sdfj\\Desktop\\mnist")#x_train为矩阵图，y_train为标签
    x_train=x_train/255
    for count in tqdm(range(0,len(x_train))):
       # view_bar(count,len(x_train))
        
#         x_train[count] = x_train[count]-x_train[count].min()
#         x_train[count] =x_train[count]/ x_train[count].max()

        hi=deal(x_train[count],w1,h_num,x_num,b1)
        ho=sigmoid(hi)
        yi=deal(ho,w2,y_num,h_num,b2)
        yo=sigmoid(yi)
#         eta_new=eta_fun(count)
        w2,b2=back2(w2,y_train[count],ho,yo,b2)
        w1,b1=back1(w1,w2,y_train[count],x_train[count],ho,yo,b1)
        if(count%99==0):
            E,C=lost_target(y_train[count],yo)
            paint_E.append(E)
            paint_C.append(C)
    x_x,y_x=read_xy()
    x_x=np.array(x_x)
    y_x=np.array(y_x)
    x_x=x_x/255
#     本数据集误差较大，先不跑   
#     for count in tqdm(range(0,len(x_x))):
#         #view_bar(count,len(x_x))
#         hi=deal(x_x[count],w1,h_num,x_num,b1)
#         ho=sigmoid(hi)
#         yi=deal(ho,w2,y_num,h_num,b2)
#         yo=sigmoid(yi)
#         w2,b2=back2(w2,y_x[count],ho,yo,b2)
#         w1,b1=back1(w1,w2,y_x[count],x_x[count],ho,yo,b1)

    for i in tqdm(range(100)):
        x_f,y_f=read_fzy()
        x_f=np.array(x_f)
        y_f=np.array(y_f)
        x_f=x_f/255
        for count in (range(0,len(x_f))):
            #view_bar(count,len(x_f))
            hi=deal(x_f[count],w1,h_num,x_num,b1)
            ho=sigmoid(hi)
            yi=deal(ho,w2,y_num,h_num,b2)
            yo=sigmoid(yi)
            w2,b2=back2(w2,y_f[count],ho,yo,b2)
            w1,b1=back1(w1,w2,y_f[count],x_f[count],ho,yo,b1) 

        x_l,y_l=read_lc()
        x_l=np.array(x_l)
        y_l=np.array(y_l)
        x_l=x_l/255
        for count in (range(0,len(x_l))):
           # view_bar(count,len(x_l))
            hi=deal(x_l[count],w1,h_num,x_num,b1)
            ho=sigmoid(hi)
            yi=deal(ho,w2,y_num,h_num,b2)
            yo=sigmoid(yi)
            w2,b2=back2(w2,y_l[count],ho,yo,b2)
            w1,b1=back1(w1,w2,y_l[count],x_l[count],ho,yo,b1)

    np.savetxt('w1.txt',w1)
    np.savetxt('w2.txt',w2)
    np.savetxt('b1.txt',b1)
    np.savetxt('b2.txt',b2)

else:
    w1=np.loadtxt('w1.txt')
    w2=np.loadtxt('w2.txt')
    b1=np.loadtxt('b1.txt')
    b2=np.loadtxt('b2.txt')


# In[13]:

#画损失函数
#pl.plot(paint_E)
#pl.xlim(0.0, 600.0)
#pl.show()


# In[14]:


#pl.plot(paint_C)
#pl.xlim(0.0, 600.0)
#pl.show()


# In[15]:

# mnist测试集的成功率测试
# x_test,y_test=load_mnist("C:\\Users\\sdfj\\Desktop\\mnist","t10k")
# x_test=x_test/255
# for count2 in range(0,len(x_test)):

#     hi=deal(x_test[count2],w1,h_num,x_num,b1)
#     ho=sigmoid(hi)
#     yi=deal(ho,w2,y_num,h_num,b2)
#     yo=sigmoid(yi)
#     get_label=yo.index(max(yo))

#     rw[y_test[count2]][get_label] += 1
        
#     if(get_label==y_test[count2]):
#         right+=1
#     else:
#         wrong+=1
#     view_bar(count2,len(x_test))
# print("\n") 
# print("测试集正确率为：",right/(wrong+right)*100,"%")  
# print("\n") 


# In[16]:


line_in=28*[0]#用于上下平移时填充

def column_move(x,example):
    """用于28✖28列的平移, 向左为负向右为正"""
    for i in range(0,28):#按行遍历
        if x<0 :
            for count in range(0,-x):
                example[i].pop(0)#删除每一行第一个
                example[i].append(0)#删除后在行尾加一个
        else:
            for count in range(0,x):
                example[i].pop(27)#删除每一行最后个0,
                example[i].insert(0,0)#删除后在行首加一个0,
    return example

def line_move(y,example):
    """用于28✖28行的平移，向上为正向下为负"""
    if y<0:
        for count in range(0,-y):
            example.pop(27)
            example.insert(0,line_in)
    else:
        for count in range(0,y):
            example.pop(0)
            example.append(line_in)
    return example

def found_pos(example):
    """寻找最佳位置"""
    only_once=0
    for l in range(0,28):
        if l>0:
            pre_num=example[l-1].count(0)
        else:
            pre_num=28

        num=example[l].count(0)
        if(num<28 and pre_num==28 and only_once==0):
            line_min=l
            only_once=1
        if(num==28 and pre_num<28):
            line_max=l-1
    if example[27].count(0)<28:
        line_max=27
        
    only_once=0

    example=[[row[i]for row in example]for i in range(len(example[0]))]

    for c in range(0,28):

        if c>0:
            pre_num=example[c-1].count(0)
        else:
            pre_num=28
        num=example[c].count(0)
        if(num<28 and pre_num==28 and only_once==0):
            c_min=c
            only_once=1

        if(num==28 and pre_num<28):
            c_max=c-1

    if example[27].count(0)<28:
        c_max=27
    
    return (line_max+line_min)//2 , (c_max+c_min)//2


# In[17]:


def mouse_event(event, x, y, flags, param):
    """鼠标事件"""
    global start, drawing, mode

    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.circle(img, (x, y), 25, (255, 255, 255), -1)
            else:
                cv2.circle(img,(x,y),30,(0,0,0),-1)
        # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.circle(img, (x, y), 25, (255, 255, 255), -1)        
        else:
            cv2.circle(img,(x,y),30,(0,0,0),-1)


# In[ ]:


def sound(label):
    """播放声音"""
    for s in label:
        if (s==0 or s==9):
            return
     
        pygame.mixer.init()
        track=pygame.mixer.music.load(str(s)+".mp3")
        pygame.mixer.music.play()
        time.sleep(0.8)
        
def cut(img):
    """切割"""
    kb = np.zeros((size,size))
    sum_line=img.sum(axis=0)
    start=False
    i_count=0
    list1=[0]
    for i in range(len(sum_line)):
            if (sum_line[i]!=0):
                    start=True#开始有子
            if(start and sum_line[i]==0):
                    i_count+=1
                    blank=i#记录空白开始的地方
            if(start and sum_line[i]!=0 and i_count<=7):#误判，参数归零
                    i_count=0
                    blank=0
            if(start and sum_line[i]!=0 and i_count>7):#到边界
                    list1.append((blank+i)//2-4)
                    i_count=0
                    blank=0
    img = img.T
    result_img = []
    for i in range(len(list1)-1):
        cut_img = np.hstack((kb[0:list1[i]].T,img[list1[i]:list1[i+1]].T,kb[list1[i+1]:].T))
        result_img.append(cut_img)
    tmp = list1.pop()
    cut_img = np.hstack((kb[0:tmp].T,img[tmp:].T))
    result_img.append(cut_img)
    return len(result_img),result_img



#a=input("1.导入图片\n2.手写图片\n")
a='2'#文件导入适配问题，暂隐
if(a=='1'):
    path=input("请输入绝对路径")
    im=Image.open(path)
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data)
    data = np.reshape(data,(28,28))
    data=255-data
    data=data.tolist()
    

    y,x=found_pos(data)
    if(x<9 or x>19):
        data=column_move(14-x,data)
    if(y<9 or y>19):
        data=line_move(y-14,data)
    data=np.matrix(data)



    #new_im = Image.fromarray(data)
    #cv2.imwrite('C:\\Users\\sdfj\\Desktop\\testyy\\3.png',new_im)
    
    data = np.reshape(data,(1,784))
    data=data/255
    hi=deal(data,w1,h_num,x_num,b1)
    ho=sigmoid(hi)
    yi=deal(ho,w2,y_num,h_num,b2)
    yo=sigmoid(yi)
    get_label=yo.index(max(yo))
    print(get_label)
    
elif(a=='2'):#画板输入
    img = np.zeros((size, size, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)
    while(True):
        cv2.imshow('image', img)
        if cv2.waitKey(1)==ord('e'):
            mode=not mode
        elif cv2.waitKey(1) == 27:
            break
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('C:\\Users\\sdfj\\Desktop\\testyy\\1.png',img)
    #img=cv2.imread('C:\\Users\\sdfj\\Desktop\\testyy\\1.png',0)
    p_n,pic=cut(img)
    out_list=[]
    for p in range(p_n):
        
        tempimg = cv2.resize(pic[p],(28,28),3)
        
        #cv2.imwrite('C:\\Users\\sdfj\\Desktop\\testyy\\2.png',tempimg)
        #im=Image.open('C:\\Users\\sdfj\\Desktop\\testyy\\2.png')
        #im = im.convert("L")    
        #data = im.getdata()
        
        data = np.matrix(tempimg)
        #new_im = Image.fromarray(data)
        #new_im.show()
        data = np.reshape(data,(28,28))
        data=data.tolist()

        y,x=found_pos(data)
        if(x<12 or x>17):
            data=column_move(14-x,data)
        if(y<12 or y>17):
            data=line_move(y-14,data)
        data=np.matrix(data)
        
        #new_im = Image.fromarray(data)
        #new_im.show()

        data = np.reshape(data,(1,784))
        data=data/255
        hi=deal(data,w1,h_num,x_num,b1)
        ho=sigmoid(hi)
        yi=deal(ho,w2,y_num,h_num,b2)
        yo=sigmoid(yi)
        get_label=yo.index(max(yo))
        print(get_label,end='')
        out_list.append(get_label)
    sound(out_list)
    

