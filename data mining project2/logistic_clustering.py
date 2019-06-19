# -*- coding: utf-8 -*-

#  DO NOT CHANGE THIS PART!!!
#  DO NOT USE PACKAGES EXCEPT FOR THE PACKAGES THAT ARE IMPORTED BELOW!!!
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import matplotlib.pyplot as plt

data=pd.read_csv('https://drive.google.com/uc?export=download&id=1AoCh22pmLHhdQtYdYUAJJqOCwF9obgVO', sep='\t')
data['class']=(data['class']=='g')*1

X=data.drop('class',axis=1).values
y=data['class'].values

trainX,testX,trainY,testY=train_test_split(X,y,stratify=y,test_size=0.2,random_state=11)

#TODO: Logistic regression
#TODO: B - Predict output based on probability of class g (y=1)
log=LogisticRegression()
log.fit(trainX,trainY)
print(log.coef_)
print(log.intercept_)
predict = log.predict_proba(testX) #probability of X
g=predict[:,1]
g #(y=1)

for i in range(19020) :  # predict ouput based on probability g
    if g[i]>=0.5 : 
        g[i]=1
        continue
    else :
        g[i]=0
        continue

#TODO: Draw a line plot (x=cutoff, y=accuracy)

cutoff=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
t_accuracy=[]
for i in range(len(cutoff)):
    ac_1=0
    ac_0=0
    
    for j in range(len(predict[:,0])):
        if predict[j,1]>cutoff[i] and testY[j]==1:
            ac_1=ac_1+1
        if predict[j,1]<cutoff[i] and testY[j]==0:
            ac_0=ac_0+1
    accuracy=ac_0+ac_1
    accuracy1=accuracy/len(testY)
    t_accuracy.append(accuracy1)        
        
    
plt.plot(cutoff, t_accuracy)



#TODO: Bernoulli naïve Bayes
#TODO: B - Estimate parameters of Bernoulli naïve Bayes
# Write user-defined function to estimate parameters of Bernoulli naïve Bayes
def BNB(X,y): 

    pmatrix=[[]]
    y_1=np.count_nonzero(y)
    y_0=len(y)-y_1
    abc=[]
    for i in range(len(X[0,])): #열
        count=0
        count1=0
        for j in range(len(X[:,0])): #행
            if X[j,i]>np.mean(X[:,i]):
                if y[j]==1:
                    count+=1
                else:
                    count1+=1
        
        ab=[count/y_1,count1/y_0]
        abc.append(ab)
    b=np.array(abc)
    pmatrix = b
    return pmatrix        

    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
            
    # TODO: Bernoulli NB
  
    



BNB(trainX,trainY)
BNB(testX,testY)
#TODO: calculate p values of several Bernoulli distributions
BNBtr=BNB(trainX,trainY)

pv=[]
pvv=[]
for i in range(len (trainX[0,])): #열개수
    k=0
        
    for j in range (len(trainX[:,0])): #행개수
           
        if trainX[j,i]==1 and trainY[j]==0:
            k=k+1
          
    pv=[k/len(X[:,0])]
    pvv.append(pv)
    
pvvv=np.array(pvv)  
p_value =np.array(pvvv) #유의수준 구하기인가? 1종오류?



#TODO: C - Predict output based on probability of class g (y=1)
#TODO: Draw a line plot (x=cutoff, y=accuracy)
def bi(X):
    for i in range(len(X[0,])):
        for j in range(len(X[:,0])):
            if X[j,i]>np.mean(X[:,i]):
                X[j,i]=1
            else:
                X[j,i]=0
    return X

bi(trainX)




evi=[]
a=1
for j in range(len(trainX[0,])): #열개수
    a*=np.count_nonzero(trainX[:,j])/len(trainX[:,j])

evidence=a

#평균보다 클확률계산
prior=np.count_nonzero(trainY)/len(trainY)
        #사전 probability

#likelihood를 구해야됨 pij를 이용해서 testX의..
#post=prior_array*likelihood
BNB(trainX,trainY)
BNB(testX,testY)

likelihood=[]
for i in range(len(testX[:,0])): #행개수
    lik=1
    for j in range(len(testX[0,])):#열개수
        
        lik=lik*((BNBtr[j,0]**testX[i,j])*((1-BNBtr[j,0])**(1-testX[i,j]))) #우도구하기
        
    likelihood.append(lik)
likelihood_array=np.array(likelihood) 
post=likelihood_array*prior/evidence
#사후확률구하기





b_accuracy=[] #정확도 구하기
for i in range(len(cutoff)):
    check=np.array(post)
    k=0
    for j in range(len(post)):
        if post[j]>cutoff[i]:
            check[j]=1
        else:
            check[j]=0
        if check[j]==testY[j]:
            k=k+1
        
    acc=k/len(post)
    b_accuracy.append(acc)
    
plt.plot(cutoff, b_accuracy)


#TODO: Nearest neighbor 
#TODO: B - k-NN with uniform weights
# Write user-deinfed function of k-NN 
# Use imported distance functions implemented by sklearn
def euclidean_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Euclidean distance between a and b

    # TODO: Euclidean distance
    d = 0
    a=np.array(a)
    b=np.array(b)
    
    d = np.sqrt(np.sum((a-b)**2))
    return d
euclidean_dist(([1,2],[2,1]),([2,1],[4,5]))
def manhattan_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Manhattan distance between a and b
    a=np.array(a)
    b=np.array(b)
    # TODO: Manhattan distance
    d = 0
    d= np.sum(np.abs(a-b))
    return d
manhattan_dist(([1,2],[2,1]),([2,1],[4,5]))


def knn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: k-NN classification
    k=k
    y_pre = []
    dist=euclidean_distances(trainX,testX)
    for i in range(len(dist[0,])):
        nei=[]
        nei_ind=[]
        list_dist=dist[:,i]
        
        nei.append(min(list_dist))
        nei_in=list_dist.argsort()    
        for j in range(k):
            nei_i=nei_in[j]
            nei_ind.append(nei_i)
 
            
        nei_array=np.array(nei_ind)
        trY=trainY[nei_array]
        if np.count_nonzero(trY)>k/2:
            pred=1
        if np.count_nonzero(trY)<k/2:
            pred=0
        y_pre.append(pred)
    y_pred=np.array(y_pre)
    
    return y_pred


k3_eu=knn(trainX,trainY,testX,3)
k5_eu=knn(trainX,trainY,testX,5)    
k7_eu=knn(trainX,trainY,testX,7)

def knn_ma(trainX,trainY,testX,k,dista=manhattan_dist):
    y_pre = []
    dista=manhattan_distances(trainX,testX)
    for i in range(len(dista[0,])):

        nei_ind=[]
        list_dist=(dista[:,i].tolist())
        nei_in=list_dist.argsort()    
        for j in range(k):
            nei_ind.append(nei_in(k))
            
            
        nei_array=np.array(nei_ind)
        trY=trainY[nei_array]
        if np.count_nonzero(trY)>k/2:
            pred=1
        else:
            pred=0
        y_pre.append(pred)
    y_pred=np.array(y_pre)
    
    return y_pred

k3_ma=knn_ma(trainX,trainY,testX,3)
k5_ma=knn_ma(trainX,trainY,testX,5)    
k7_ma=knn_ma(trainX,trainY,testX,7)


# TODO: Calculate accuracy of test set 
#       with varying the number neareset neighbors (k) and distance metrics
#       using k-NN
def accu(a,b):
    count=0
    for i in range(len(a)):
        
        if a[i]==b[i]:
            count=count+1
    
    accuracy=count/len(a)
    
    return accuracy
    
    
    
accu(k3_eu,testY)
accu(k5_eu,testY)
accu(k7_eu,testY)
accu(k3_ma,testY)
accu(k5_ma,testY)
accu(k7_ma,testY)


#TODO: C - weighted k-NN
# Write user-deinfed function of weighted k-NN
# Use imported distance functions implemented by sklearn
def wknn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## Weighted K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: weighted k-NN classification
    y_pred = []
    y_pre = []
    
    dist=euclidean_distances(trainX,testX)
    for i in range(len(dist[0,])):
        nei=[]
        re_we=[]
        nei_ind=[]
        we_sum=0
        weight_g=0
        weight_h=0
        list_dist=(dist[:,i].tolist())
        for j in range(k):
            nei.append(min(list_dist)) #이웃의 값
            nei_ind.append(list_dist.index(min(list_dist)))#이웃값 인덱스
            list_dist.pop(list_dist.index(min(list_dist))) #최솟값 삭제
        ne_array=np.array(nei) #이웃값 배열화
        nei_array=np.array(nei_ind) #이웃값 인덱스 배열화
        trY=trainY[nei_array] #이웃값에서 trainY값 추출
        for h in range(k):
            re_we.append(1/ne_array[h]) #역수값
        re_we_array=np.array(re_we) #역수값 배열화
        we_sum=np.sum(re_we)
        wei_g=0
        wei_h=0
        for l in range(k):
            if trY[l]==1:
                wei_g=wei_g+re_we_array[l]
            if trY[l]==0:
                wei_h=wei_h+re_we_array[l]
        weight_g=wei_g/we_sum
        weight_h=wei_h/we_sum
        if weight_g>weight_h:
            pred=1
        if weight_g<=weight_h:
            pred=0
        
        
        y_pre.append(pred)
    y_pred=np.array(y_pre)
    
    
    
    
    
    return y_pred    

wk3_eu=wknn(trainX,trainY,testX,3)
wk5_eu=wknn(trainX,trainY,testX,5)
wk7_eu=wknn(trainX,trainY,testX,7)

def wknn_ma(trainX,trainY,testX,k,dist=manhattan_dist):#맨해튼거리 가중
    y_pred = []
    y_pre = []
    
    dist=manhattan_distances(trainX,testX)
    for i in range(len(dist[0,])):
        nei=[]
        re_we=[]
        nei_ind=[]
        we_sum=0
        weight_g=0
        weight_h=0
        list_dist=(dist[:,i].tolist())
        for j in range(k):
            nei.append(min(list_dist)) #이웃의 값
            nei_ind.append(list_dist.index(min(list_dist)))#이웃값 인덱스
            list_dist.pop(list_dist.index(min(list_dist))) #최솟값 삭제
        ne_array=np.array(nei) #이웃값 배열화
        nei_array=np.array(nei_ind) #이웃값 인덱스 배열화
        trY=trainY[nei_array] #이웃값에서 trainY값 추출
        for h in range(k):
            re_we.append(1/ne_array[h]) #역수값
        re_we_array=np.array(re_we) #역수값 배열화
        we_sum=np.sum(re_we)
        wei_g=0
        wei_h=0
        for l in range(k):
            if trY[l]==1:
                wei_g=wei_g+re_we_array[l]
            if trY[l]==0:
                wei_h=wei_h+re_we_array[l]
        weight_g=wei_g/we_sum
        weight_h=wei_h/we_sum
        if weight_g>weight_h:
            pred=1
        if weight_g<=weight_h:
            pred=0
        
        
        y_pre.append(pred)
    y_pred=np.array(y_pre)
    
    return y_pred



wk3_ma=wknn_ma(trainX,trainY,testX,3)
wk5_ma=wknn_ma(trainX,trainY,testX,5)
wk7_ma=wknn_ma(trainX,trainY,testX,7)

# TODO: Calculate accuracy of test set 
#       with varying the number neareset neighbors (k) and distance metrics
#       using weighted k-NN

def accu(a,b):
    count=0
    for i in range(len(a)):
        
        if a[i]==b[i]:
            count=count+1
    
    accuracy=count/len(a)
    
    return accuracy
    

accu(wk3_eu,testY)
accu(wk5_eu,testY)
accu(wk7_eu,testY)
accu(wk3_ma,testY)
accu(wk5_ma,testY)
accu(wk7_ma,testY)

#표준화 시키기
def standard(trainX):
    
    valid=a
    for i in range(len(valid[0,])): #열개수
        means=np.mean(valid[:,i])
        std=np.std(valid[:,i])
        for j in range(len(valid[:,0])): #행개수
            valid[j,i]=(valid[j,i]-means)/std
    
    return valid
valid_trainX=standard(trainX)
valid_testX=standard(testX)

#평균과 표준편차로 표준화시킨 값구하기
accu(knn(valid_trainX,trainY,valid_testX,3),testY) #유클리디안knn 정확도
accu(knn(valid_trainX,trainY,valid_testX,5),testY)
accu(knn(valid_trainX,trainY,valid_testX,7),testY)

a=knn_ma(valid_trainX,trainY,valid_testX,3) #맨해튼knn 정확도
b=knn_ma(valid_trainX,trainY,valid_testX,5)
c=knn_ma(valid_trainX,trainY,valid_testX,7)
accu(a,testY)
accu(b,testY)
accu(c,testY)

a1=wknn(valid_trainX,trainY,valid_testX,3) #유클리디안 weight_knn 정확도
a2=wknn(valid_trainX,trainY,valid_testX,5)
a3=wknn(valid_trainX,trainY,valid_testX,7)
accu(a1,testY)
accu(a2,testY)
accu(a3,testY)

b1=wknn_ma(valid_trainX,trainY,valid_testX,3) #맨해튼 weight_knn 정확도
b2=wknn_ma(valid_trainX,trainY,valid_testX,5)
b3=wknn_ma(valid_trainX,trainY,valid_testX,7)
accu(b1,testY)
accu(b2,testY)
accu(b3,testY)



#TODO: k-means clustering
#TODO: B - k-means clustering
# Write user-defined function of k-means clustering
def kmeans(X,k,max_iter=300):
    ############ K-MEANS CLUSTERING ##########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of clusters
    # max_iter: the maximum number of iteration
    # OUTPUT
    # label: cluster label (len(label)=n)
    # centers: cluster centers (k by p)
    ##########################################
    # If average distance between old centers and new centers is less than 0.000001, stop
    
    # TODO: k-means clustering
    label = []
    centers = []
    init_ce=[]
    group=[]
    for i in range(k): #k개수만큼 초기값지정
        init_ce.append(X[np.random.randint(0,len(X))])
    init_center=np.array(init_ce)
    
    
    ce_dist=euclidean_distances(X,init_center) #초기값기준 거리
    for i in range(len(ce_dist)):
        group.append(np.argmin(ce_dist[i,]))
    group1=np.array(group) #초기값을 기준으로 구한 군집
    
    
    
    
    for a in range(max_iter):
        #군집의 평균구하기
        second_group=[] #군집형성리스트
        distri1=[]
        for j in range(k):
            distri=[]
            
            for i in range(len(ce_dist)):
       
                if group1[i]==j:
                    distri.append(i)
                
            distri1.append(distri)
        distri2=np.array(distri1)      #군집의 모음
        
        
        
        update_ce1=[] #두번째 열에 넣을 내용
        for i in range(k):
            update_ce=[] #첫번째 열에 넣을 내용
            
            for j in range(len(X[0,])):
                update_ce.append(np.mean(X[distri2[i],j]))
            update_ce1.append(update_ce)
        update_ce2=np.array(update_ce1) #배열 완성, 평균을 기준으로 수정된 중앙값
        center_distance=euclidean_distances(X,update_ce2) #다시 거리계산
        for i in range(len(center_distance)): #군집형성
            second_group.append(np.argmin(center_distance[i,]))
        second_group1=np.array(second_group)
        
        
        
    
    label=second_group1
    centers=update_ce2
    
    
    
    return (label, centers)


# TODO: Calculate centroids of two clusters
label,centers=kmeans(X,2,max_iter=300)    




#TODO: C - homogeneity and completeness 

c0=0 #실제 값이 0인값
for i in y: 
    if i==0:
        c0=c0+1
c1=len(y)-c0


k0=0 #예상값이 0인값
for i in label:
    if i==0:
        k0=k0+1
k1=len(label)-k0




n_ck11=0 # 둘다 1인값
for i in range(len(y)):
    if y[i]==1 and label[i]==1:
        n_ck11=n_ck11+1
        
n_ck00=0#둘다 0인값
for i in range(len(y)):
    if y[i]==0 and label[i]==0:
        n_ck00=n_ck00+1
        
n_ck10=0#둘다 0인값
for i in range(len(y)):
    if y[i]==1 and label[i]==0:
        n_ck10=n_ck10+1        

n_ck01=0#둘다 0인값
for i in range(len(y)):
    if y[i]==0 and label[i]==1:
        n_ck01=n_ck01+1        
        
        
h_c= -(c0/len(y))*np.log2(c0/len(y))-(c1/len(y))*np.log2(c1/len(y)) #H(C)

h_ck= -(n_ck00/len(y))*np.log2(n_ck00/k0)-(n_ck01/len(y))*np.log2(n_ck01/k1)-(n_ck10/len(y))*np.log2(n_ck10/k0)-(n_ck11/len(y))*np.log2(n_ck11/k1)

homogeneity=1-h_ck/h_c


h_k= -(k0/len(y))*np.log2(k0/len(y))-(k1/len(y))*np.log2(k1/len(y))
h_kc= -(n_ck00/len(y))*np.log2(n_ck00/c0)-(n_ck01/len(y))*np.log2(n_ck01/c1)-(n_ck10/len(y))*np.log2(n_ck10/c0)-(n_ck11/len(y))*np.log2(n_ck11/c1)

completeness=1-h_kc/h_k













