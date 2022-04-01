import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle #save and load

mymodel = pickle.load(open("saveload_mlr_model.pickle","rb"))

dataframe = pd.read_csv("multilinearregression.csv", sep=";")
# print(dataframe)

#linear regression modeli tanımlayalım:

reg = linear_model.LinearRegression()
reg.fit(dataframe[['alan','odasayisi','binayasi']],dataframe['fiyat']) #öğrenme


#Prediction 
tahmin1 = reg.predict([[230,4,10]]) #tahmini fiyat 530..
# print(tahmin1)
model_dosyasi = "saveload_mlr_model.pickle"
pickle.dump(reg , open(model_dosyasi,'wb')) #writing and open in binary mode


tahmin2 = reg.predict([[230,6,0]]) #predict->586...
# print(tahmin2)
tahmin3 = reg.predict([[355,3,20]]) #616...
# print(tahmin3)

#toplu tahmin
toplutahmin = reg.predict([[230,4,10], [230,6,0], [355,3,20]])
# print(toplutahmin)


result1 = reg.coef_ #katsayılar
# print(result1)

result2 = reg.intercept_    #sabit değer
# print(result2)

#Multiple Linear Regression formülümüze dönersek 
#y = a + b1X1 + b2X2 + b3X3 ...

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230 
x2 = 4
x3 = 10

y = a + b1*x1 +b2*x2 + b3*x3
print(y) #530...

