# -*- coding: utf-8 -*-

# Do not change this part
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 

# Data load
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1O74eCM8zlPxCFEuEpshmFXA3HpKT1qbC')
#data=pd.read_csv('https://drive.google.com/uc?export=download&id=1YPnojmYq_2B_lrAa78r_lRy-dX_ijpCM', sep='\t')
#TODO: explanatory analysis

#TODO: B, C, D - Darw pairwise scatter plots

plt.scatter(data['Max_temperature'], data['Mean_temperature'])
plt.xlabel('Max_temperature')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Min_temperature'], data['Mean_temperature'])
plt.xlabel('Min_temperature')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Dewpoint'], data['Mean_temperature'])
plt.xlabel('Dewpoint')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Precipitation'], data['Mean_temperature'])
plt.xlabel('Precipitation')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Sea_level_pressure'], data['Mean_temperature'])
plt.xlabel('Sea_level_pressure')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Standard_pressure'], data['Mean_temperature'])
plt.xlabel('Standard_pressure')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Visibility'], data['Mean_temperature'])
plt.xlabel('Visibility')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Wind_speed'], data['Mean_temperature'])
plt.xlabel('Wind_speed')
plt.ylabel('Mean_temperature')
plt.show()
plt.scatter(data['Max_wind_speed'], data['Mean_temperature'])
plt.xlabel('Max_wind_speed')
plt.ylabel('Mean_temperature')
plt.show()

#TODO: E, F - Calculate correlation matrix

x= data.iloc[:,:]
correlation_matrix = x.corr()
print(correlation_matrix)





#corelation betweend input and output
print (data[['Max_temperature','Mean_temperature']].corr())
print (data[['Min_temperature','Mean_temperature']].corr())
print (data[['Dewpoint','Mean_temperature']].corr())
print (data[['Precipitation','Mean_temperature']].corr())
print (data[['Sea_level_pressure','Mean_temperature']].corr())
print (data[['Standard_pressure','Mean_temperature']].corr())
print (data[['Visibility','Mean_temperature']].corr())
print (data[['Wind_speed','Mean_temperature']].corr())
print (data[['Max_wind_speed','Mean_temperature']].corr())

# set input value 
a = data[['Max_temperature']]
b = data[['Min_temperature']]
c = data[['Dewpoint']]
d = data[['Precipitation']]
e = data[['Sea_level_pressure']]
f = data[['Standard_pressure']]
g = data[['Visibility']]
h = data[['Wind_speed']]
i = data[['Max_wind_speed']]

#TODO: linear regression
#TODO: B - Calculate VIF


reg = LinearRegression()
reg.fit(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']], data['Mean_temperature'])

reg.coef_
reg.intercept_

y_pred = reg.predict(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']])
# y predict는 두 변수사이의 회귀선을 측정해주는것! 즉 x변수를 하나만 두면 각각의 input을 지정가능
SSE = sum((data['Mean_temperature']-y_pred)**2)

n=len(data)
p=4
MSE = SSE/(n-p-1)

X=data[['Max_temperature','Min_temperature','Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']].values
#MSE구함

SSE
SSR=sum((y_pred-data['Mean_temperature'].mean())**2)
SSR

SST = sum((data['Mean_temperature']-data['Mean_temperature'].mean())**2)


r2 = SSR/SST

# set output value
j = data['Mean_temperature']
 
ar = LinearRegression()
ar.fit(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Max_temperature'])
ar2 = ar.score(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Max_temperature'])
a_vif = 1/(1-ar2)
a_vif
br = LinearRegression()
br.fit(data[['Max_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Min_temperature'])
br2 = br.score(data[['Max_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Min_temperature'])
b_vif = 1/ (1-br2)
b_vif
cr = LinearRegression()
cr.fit(data[['Min_temperature', 'Max_temperature','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed',]],data['Dewpoint'])
cr2 = cr.score(data[['Min_temperature', 'Max_temperature','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Dewpoint'])
c_vif = 1/(1-cr2)
c_vif
dr = LinearRegression()
dr.fit(data[['Min_temperature', 'Dewpoint','Max_temperature','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Precipitation'])
dr2 = dr.score(data[['Min_temperature', 'Dewpoint','Max_temperature','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Precipitation'])
d_vif = 1/(1-dr2)
er = LinearRegression()
er.fit(data[['Min_temperature', 'Dewpoint','Precipitation','Max_temperature','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed']],data['Sea_level_pressure'])
er2 = er.score(data[['Min_temperature', 'Dewpoint','Precipitation','Max_temperature','Standard_pressure', 'Visibility','Wind_speed','Max_wind_speed',]],data['Sea_level_pressure'])
e_vif = 1/(1-er2)
fr = LinearRegression()
fr.fit(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Max_temperature', 'Visibility','Wind_speed','Max_wind_speed']],data['Standard_pressure'])
fr2 = fr.score(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Max_temperature', 'Visibility','Wind_speed','Max_wind_speed']],data['Standard_pressure'])
f_vif = 1/(1-fr2)
gr = LinearRegression()
gr.fit(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Max_temperature','Wind_speed','Max_wind_speed']],data['Visibility'])
gr2 = gr.score(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Max_temperature','Wind_speed','Max_wind_speed']],data['Visibility'])
g_vif = 1/(1-gr2)
hr = LinearRegression()
hr.fit(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Max_temperature','Max_wind_speed']],data['Wind_speed'])
hr2 = hr.score(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Max_temperature','Max_wind_speed']],data['Wind_speed'])
h_vif = 1/(1-hr2)
h_vif = 1/(1-hr2)
ir = LinearRegression()
ir.fit(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_temperature']],data['Max_wind_speed'])
ir2 = ir.score(data[['Min_temperature', 'Dewpoint','Precipitation','Sea_level_pressure','Standard_pressure', 'Visibility','Wind_speed','Max_temperature']],data['Max_wind_speed'])
i_vif = 1/(1-ir2)
    #VIF계산



# Model 1
#TODO: C - Train a lineare regression model
#          Calculate t-test statistics 

new_reg = LinearRegression()
new_reg.fit(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']],j)
#새로운 선형회귀를 만들었고, 


new_reg_pred = new_reg.predict(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']])

new_reg.coef_
new_reg.intercept_



SSE = sum((data['Mean_temperature']-new_reg_pred)**2)
SSE

X=data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']].values
X
X = np.c_[np.ones(n),X]
X
xtx = np.matmul(X.T,X) #XTX구함
xtx

xtx_inv=np.linalg.inv(xtx) #XtX의 역행렬
xtx_inv

SSE = sum((data['Mean_temperature']-new_reg_pred)**2)
SSE

n=len(data)
p=4
MSE=SSE/(n-p-1)
MSE
#i=0은 intercept를 의미하므로 1부터
#y = b0+b1x1+b2x2+...
i=[1,2,3,4]
#표준오차
se=np.sqrt(MSE*xtx_inv[i,i])
se

#beta = 결국 coeffecient이다
beta = np.insert(new_reg.coef_,0,new_reg.intercept_)
beta
#통계량
t=beta[i]/(np.sqrt(MSE*xtx_inv[i,i]))
t


alpha =0.05
#p-value = 통계량 보다 큰 쪽의 넓이
1-tdist.cdf(np.abs(t), n-p-1)
#p-value 구하기, 확률밀도함수에서 t값보다 높은 부분=p-value --넓이
tdist.ppf(1-alpha/2,n-p-1)
#정규분포 에서 1-alpha/2만큼의 넓이를 가지고있는 값을 반환 --x 값
np.abs(t)


#TODO: D - Calculate adjusted R^2
n=len(data)
p=4
new_r2 = new_reg.score(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']],data['Mean_temperature'])
new_r2

adj_r2=1-((n-1)/(n-p-1))*(1-new_r2)
adj_r2

#TODO: E - Apply F-test

SSE = sum((new_reg_pred-data['Mean_temperature'])**2)
SST = sum((data['Mean_temperature']-np.mean(data['Mean_temperature']))**2)
SSR = sum((new_reg_pred-np.mean(data['Mean_temperature']))**2)

MSE = SSE/(n-p-1)
MSR = SSR/p
MSE
MSR

f=MSR/MSE
f
1-fdist.cdf(f,p,n-p-1)

# Model 2
#TODO: F - Check the appropriateness of the given set of variables

sec_reg = LinearRegression()
sec_reg.fit(data[['Visibility','Dewpoint','Precipitation']],j)
sec_reg_pred = sec_reg.predict(data[['Visibility','Dewpoint','Precipitation']])

SSE1= sum((sec_reg_pred-j)**2)
SST1= sum((j-np.mean(j))**2)
SSR1= sum((sec_reg_pred-np.mean(j))**2)
plt.scatter(new_reg_pred,j)
plt.scatter(sec_reg_pred,j)

ar=LinearRegression()
ar.fit(data[['Dewpoint','Precipitation']],data['Visibility'],j)
Vi_r2=ar.score(data[['Dewpoint','Precipitation']],data['Visibility'])
Vi_r2
Vi_vif=1/(1-Vi_r2)
Vi_vif

br=LinearRegression()
br.fit(data[['Visibility','Precipitation']],data['Dewpoint'])
Dew_r2=br.score(data[['Visibility','Precipitation']],data['Dewpoint'])
Dew_r2
Dew_vif=1/(1-Dew_r2)
Dew_vif

cr=LinearRegression()
cr.fit(data[['Visibility','Dewpoint']],data['Precipitation'])
Pre_r2=cr.score(data[['Visibility','Dewpoint']],data['Precipitation'])
Pre_r2
Pre_vif=1/(1-Pre_r2)
Pre_vif

Vi_vif
Dew_vif
Pre_vif
#vif는 예측변수간의 상관관계를 나타내며, 이를 통해 이 변수가 신뢰성이 있는지 없는지 판단이 가능.. 높으면 예측변수간의 상관관계가 높아서 신뢰성 하락

#TODO: G - Train a lineare regression model
# Calculate t-test statistics 


sec_reg.coef_
sec_reg.intercept_

X=data[['Visibility','Dewpoint','Precipitation']].values
X
X = np.c_[np.ones(n),X]
X
xtx = np.matmul(X.T,X) #XTX구함
xtx

xtx_inv=np.linalg.inv(xtx) #XtX의 역행렬
xtx_inv


n=len(data)
n
p=3
MSE1=SSE1/(n-p-1)
MSE1
#i=0은 intercept를 의미하므로 1부터
#y = b0+b1x1+b2x2+...
i=[1,2,3]
#표준오차
se=np.sqrt(MSE1*xtx_inv[i,i])
se

#beta = 결국 coeffecient이다
beta = np.insert(sec_reg.coef_,0,sec_reg.intercept_)
beta
#통계량
t=beta[i]/(np.sqrt(MSE1*xtx_inv[i,i]))
t

np.abs(t)
alpha =0.05
#p-value = 통계량 보다 큰 쪽의 넓이
1-tdist.cdf(np.abs(t), n-p-1)
#p-value 구하기, 확률밀도함수에서 t값보다 높은 부분=p-value --넓이
tdist.ppf(1-alpha/2,n-p-1)
#정규분포 에서 1-alpha/2만큼의 넓이를 가지고있는 값을 반환 --x 값




#TODO: H - Calculate adjusted R^2

sec_r2 = new_reg.score(data[['Visibility','Dewpoint','Precipitation']],data['Mean_temperature'])
sec_r2

adj_r2=1-((n-1)/(n-p-1))*(1-new_r2)
adj_r2



#TODO: I - Test normality of residuals

j=data['Mean_temperature']
n=len(data)
p=3
resi = (j-sec_reg_pred)
plt.hist(resi,range(-20,20))

chi_square = sum((((sec_reg_pred-np.mean(j))**2))/np.mean(j))

e2 = resi-np.mean(resi)

S = ((1/n)*((sum((e2)**3))))/(((1/n)*(sum((e2)**2)))**(3/2))
C = ((1/n)*((sum((e2)**4))))/(((1/n)*(sum((e2)**2)))**(4/2))
JB = ((n-p)/6)*((S**2)+(1/4)*((C-3)**2))
JB

chi2.ppf(1-0.05,2)

1-chi2.cdf(JB,2)


#TODO: J - Test homoskedasticty of residuals

reg3 = LinearRegression()
reg3.fit(data[['Visibility','Dewpoint','Precipitation']],resi**2)

reg3.pred=reg3.predict(data[['Visibility','Dewpoint','Precipitation']])


LM = n*sec_reg.score(data[['Visibility','Dewpoint','Precipitation']],resi**2)

LM
1-chi2.cdf(LM,p)

plt.scatter(range(1,len(data)+1),resi)
