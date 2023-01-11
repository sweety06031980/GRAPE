
# coding: utf-8

# In[20]:


import pandas as pd


# In[ ]:


df=pd.read_excel("soil.xlsx")
df.isna().sum()


# In[3]:


df['Area (Acres)'].fillna(0,inplace=True)
df.dropna(inplace=True)


# In[4]:


df


# In[5]:


v=df['Village'].unique()


# In[21]:


df['Variety'].unique()


# In[22]:


primary=["pH","P","K","S","N","NH4-N","EC","CaCo3","Ca","OC","Mg"]
#pH,N,P,K
secondary=["Fe","Mn","Zn","Cu","B","HCO3","Cl"]


# In[23]:


p=[]
for i in primary[0:6]:
    p.append(round(df[df['Village']=="Pimpalas Ramche"][i].mean(),2))


# In[24]:


s=[]
for i in secondary[0:6]:
    s.append(round(df[df['Village']=="Pimpalas Ramche"][i].mean(),2))


# In[25]:


# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = primary[0:6]

data = p

# Creating plot
fig = plt.figure(figsize =(9,17))
plt.pie(data, labels = cars,autopct='%1.1f%%')
plt.title('Macronutrients of soils')
# show plot
plt.show()


# In[11]:


# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = secondary[0:6]

data = s

# Creating plot
fig = plt.figure(figsize =(7,10))
plt.pie(data, labels = cars,autopct='%1.1f%%')
plt.title('Micronutrients of soils')
# show plot
plt.show()


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
final=[]
for i in v:
    col=list(df.columns)
    cor=[]
    for p in primary:
        if p in col:
            f=[]
            for s in secondary:
                r=df.loc[df.Village==i]
                #r=r.iloc[::,9:]
                c=r[p].corr(r[s]).round(2)
                f.append(c)
            f.insert(0,p)
            f.insert(0,i)
            cor.append(f)
    final.append(cor)
secondary.insert(0,"Villages")
secondary.insert(1,"Chemical properties Available nutrients")

    #cor.insert(0,i)   
    #print(cor)
    #plt.figure(figsize=(10,10))
    #plot=sns.heatmap(r.corr().round(2),annot=True)
#final


# In[27]:


fd=[]
for k in final:
    dp=pd.DataFrame(k)
    dp.columns=secondary
    fd.append(dp)
cd=pd.concat(fd)
cd


#dp.style.background_gradient(cmap='coolwarm')
#cd.to_excel("Correlation_of_soil_properties.xlsx",index=False)
#cd['Chemical properties Available nutrients'=='N']]]


# In[80]:


dp=pd.DataFrame(final[0])
dp.columns=secondary
dp.style.background_gradient(cmap='coolwarm')


# In[48]:



cd['Fe'].fillna(cd['Fe'].mean(),inplace=True)
cd['Mn'].fillna(cd['Mn'].mean(),inplace=True)
cd['Zn'].fillna(cd['Zn'].mean(),inplace=True)
cd['Cu'].fillna(cd['Cu'].mean(),inplace=True)
cd['B'].fillna(cd['B'].mean(),inplace=True)
cd['HCO3'].fillna(cd['HCO3'].mean(),inplace=True)
cd['Cl'].fillna(cd['Cl'].mean(),inplace=True)
cd.isna().sum()
cd.head()


# In[29]:


for i in range(5):
    cor=cd.iloc[i,2:].values.tolist()
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i,y[i],y[i])

    if __name__ == '__main__':
    # creating data on which bar chart will be plot
        x = ["Fe","Mn","Zn","Cu","B","HCO3","Cl"]
        y = cor
        print(y)

        # making the bar chart on the data
        plt.bar(x, y)

        # calling the function to add value labels
        addlabels(x, y)

        # giving title to the plot
        plt.title("Corellation B/W Primary and Seconday Nutrients")

        # giving X and Y labels
        plt.xlabel(f"Primary Nutrient({primary[i]})")
        plt.ylabel("Seconday Nutrients")

        # visualizing the plot
        plt.show()


# # Categorization of soil parameters and nutrients

# In[81]:


n=df.iloc[::,9:].columns

cpn=[]
for vl in v:
    high=[]
    low=[]
    medium=[]

    for i in n:
        r=df.loc[df.Village==vl]
        per_h=r[i].sum()/len(r)
        high.append(round(r[i].sum()/len(r),2))
        low.append(round(per_h-1.0,2))
        medium.append(str(round(per_h-1.0,2))+"-"+str(round(r[i].sum()/len(r),2)))
    cs=pd.DataFrame({"Village":vl,"Parameter":n,"LOW (Acidic)":low,"MEDIUM (Neutral)":medium,"HIGH (Alkaline)":high})
    cpn.append(cs)
pn=pd.concat(cpn)
pn


# In[77]:


cdf=[]
for cv in v:
    pr=['pH','N','P','K']
    m=df.loc[df.Village==cv]
    cp=[]
    for p in pr:
        mn=round(m[p].mean(),2)
        r=str(m[p].min())+"-"+str(m[p].max())
        cp.append([cv,p,r,mn])
    d2=pd.DataFrame(cp,columns=['Village','Chemical Property','Range',"Mean"])
    cdf.append(d2)
cs2=df.iloc[::,0:].columns
cs2=pd.concat(cdf)
cs2


# # Nutrient properties of soils with mean values

# In[56]:


cdf=[]
for cv in v:
    pr=['pH','N','P','K']
    m=df.loc[df.Village==cv]
    cp=[]
    for p in pr:
        mn=round(m[p].mean(),2)
        r=str(m[p].min())+"-"+str(m[p].max())
        cp.append([cv,p,r,mn])
    d2=pd.DataFrame(cp,columns=['Village','Chemical Property','Range',"Mean"])
    cdf.append(d2)
cs2=pd.concat(cdf)
cs2


# In[ ]:


# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = cs2.iloc[0:4,1].values.tolist()

data = cs2.iloc[0:4,3].values.tolist()

# Creating plot
fig = plt.figure(figsize =(5, 7))
plt.pie(data, labels = cars,autopct='%1.1f%%')
plt.title('Nutrient  properties of soils')
# show plot
plt.show()


# In[36]:


cdf2=[]
for cv in v:
    pr1=['Cu','Zn','Fe','Mn','B']
    m1=df.loc[df.Village==cv]
    cp=[]
    for p in pr1:
        mn=round(m1[p].mean(),2)
        r=str(m1[p].min())+"-"+str(m1[p].max())
        cp.append([cv,p,r,mn])
    d2=pd.DataFrame(cp,columns=['Village','Chemical Property','Range',"Mean"])
    cdf2.append(d2)
cs2=pd.concat(cdf2)
cs2


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = cs2.iloc[0:4,1].values.tolist()

data = cs2.iloc[0:4,3].values.tolist()

# Creating plot
fig = plt.figure(figsize =(5, 7))
plt.pie(data, labels = cars,autopct='%1.1f%%')
plt.title('Nutrient properties of soils')
# show plot
plt.show()


# # Percentage samples deficient micronutrients in soils

# In[37]:


cdf2=[]
for cv in v:
    pr1=['Cu','Zn','Fe','Mn','B']
    m1=df.loc[df.Village==cv]
    cp=[]
    for p in pr1:
        mn=round(sum(m1[p])/len(df)*100,2)
        cp.append([cv,p,mn])
    d2=pd.DataFrame(cp,columns=['Village','Chemical Property',"Percentage"])
    cdf2.append(d2)
cs2=pd.concat(cdf2)
cs2


# In[38]:


from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = cs2.iloc[0:4,1].values.tolist()

data = cs2.iloc[0:4,2].values.tolist()

# Creating plot
fig = plt.figure(figsize =(5, 7))
plt.pie(data, labels = cars,autopct='%1.1f%%')
plt.title('Percentage samples deficient micronutrients in soils')
# show plot
plt.show()


# In[39]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
labelencoder_Y = LabelEncoder()
df['Y']= labelencoder_Y.fit_transform(df.iloc[:,3].values)
x = df.iloc[:, 9:28].values 
Y = df["Y"].values 
X=[]
for q in x:
    X.append([float(i) for i in q])
#Split the data again, but this time into 75% training and 25% testing data sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    #Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def models(X_train,Y_train):
    
    #Using Logistic Regression 
      from sklearn.linear_model import LogisticRegression
      log = LogisticRegression(random_state = 0)
      log.fit(X_train, Y_train)

      #Using KNeighborsClassifier 
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
      knn.fit(X_train, Y_train)

      #Using SVC linear
      from sklearn.svm import SVC
      svc_lin = SVC(kernel = 'linear', random_state = 0)
      svc_lin.fit(X_train, Y_train)

      #Using SVC rbf
      from sklearn.svm import SVC
      svc_rbf = SVC(kernel = 'rbf', random_state = 0)
      svc_rbf.fit(X_train, Y_train)

      #Using GaussianNB 
      from sklearn.naive_bayes import GaussianNB
      gauss = GaussianNB()
      gauss.fit(X_train, Y_train)

      #Using DecisionTreeClassifier 
      from sklearn.tree import DecisionTreeClassifier
      tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
      tree.fit(X_train, Y_train)

      #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
      from sklearn.ensemble import RandomForestClassifier
      forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
      forest.fit(X_train, Y_train)

      #print model accuracy on the training data.
      print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
      print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
      print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
      print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
      print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
      print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
      print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

      return  log, knn, svc_lin, svc_rbf, gauss, tree, forest
model = models(X_train,Y_train)
    #Print Prediction of KNN model
pred_k = model[5].predict(X_test)
print('Prediction of KNN model',pred_k)
    

    


# In[44]:


alg_model=pd.DataFrame({"Logistic Regression":model[0].predict(X_test),"KNN":model[1].predict(X_test),"SVM(L)":model[2].predict(X_test),"SVM(RBF)":model[3].predict(X_test),"GNB":model[4].predict(X_test),"DT":model[5].predict(X_test),"Random Forest":model[6].predict(X_test)})
a=round(alg_model.mean(),2)
av=a.values.tolist()
ac=list(alg_model.columns)


# In[41]:


import pandas as pd
from matplotlib import pyplot as plt




# Figure Size
fig = plt.figure(figsize =(10, 7))

# Horizontal Bar Plot
plt.bar(ac, av)

# Show Plot
plt.show()


# In[46]:


s=set(Y)
v1=df['Village'].values
data_dict=dict(zip(list(s), list(set(v1))))
z=[data_dict[w] for w in list(set(pred_k))]
result=pd.DataFrame({"Village":z,"Soil_Status":"Fertile"})


# In[45]:


cars = ['Fertile','Not Fertile']

data = [len(result),len(data_dict)-len(result)]

# Creating plot
fig = plt.figure(figsize =(5, 7))
plt.pie(data, labels = cars,autopct='%1.1f%%')
plt.title('Percent soils Feritility')
# show plot
plt.show()
print(result)

