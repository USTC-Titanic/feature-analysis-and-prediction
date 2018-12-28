
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
from collections import Counter
import matplotlib.pyplot as plt
train=pd.read_csv('./desktop/all/train.csv')#读入训练数据集
test=pd.read_csv('./desktop/all/test.csv')#读入测试数据集
print('训练数据集规模',train.shape)
print('测试数据集规模',test.shape)


# In[2]:


df=train.append(test,sort=True) #合并数据集，方便进行数据预处理


# In[3]:


print('合并后的数据集规模',df.shape)


# In[4]:


pd.options.display.float_format='{:,.3f}'.format #数据显示模式
df.describe()#查看数据的基本统计信息


# In[5]:


df.info()#查看每一列的数据以及数据类型，进行缺失值分析


# In[6]:


df.head(10)


# In[7]:


df['Age']=df['Age'].fillna(df['Age'].mean())#年龄缺失值用均值填充


# In[8]:


train['Cabin']=train['Cabin'].map(lambda x:'known' if type(x)==str else 'unknown')
sns.countplot(x='Cabin',hue="Survived",data=train)


# In[9]:


df['Cabin']=df['Cabin'].fillna('U')#Cbain缺失值用‘U’填充，代表Unknown
df['Fare']=df['Fare'].fillna(df['Fare'].mean())#Fare缺失值用均值填充
print(Counter(df['Embarked']))#Embarked的类别有三类：S、C、Q，众数‘S'


# In[10]:


sns.countplot(x='Embarked',hue="Survived",data=train)


# In[11]:


df['Embarked']=df['Embarked'].fillna('S')#Embarked缺失值用众数'S'填充


# In[12]:


#查看缺失值处理之后的数据
df.info()


# In[13]:


#年龄特征分类
train['Age']=train['Age'].map(lambda x: 'child' if x<=16 else 'youth' if 16<x<=32 else 'adlut' if 32<x<=48 else 'old'  if 48<x<=64 else 'too old' )
sns.countplot(x='Age',hue="Survived",data=train)


# In[14]:


AgeDf=pd.DataFrame()
AgeDf['child']=df['Age'].map(lambda x:1 if x<=16 else 0)
AgeDf['youth']=df['Age'].map(lambda x:1 if 16<x<=32 else 0)
AgeDf['adult']=df['Age'].map(lambda x:1 if 32<x<=48 else 0)
AgeDf['old']=df['Age'].map(lambda x:1 if 48<x<=64 else 0)
AgeDf['too old']=df['Age'].map(lambda x:1 if x>64 else 0)
AgeDf.head()


# In[15]:


df=pd.concat([df,AgeDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
df.drop('Age',axis=1,inplace=True)#删除原来的Age属性
df.head(5)


# In[16]:


cabinDf=pd.DataFrame()#创建pandas中的DataFrame对象
df['Cabin']=df['Cabin'].map(lambda x:x[0])
cabinDf=pd.get_dummies(df['Cabin'],prefix='Cabin')#使用get_dummies进行oone-hot编码，列名前缀是Cabin
cabinDf.head()


# In[17]:


df=pd.concat([df,cabinDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
df.drop('Cabin',axis=1,inplace=True)#删除原来的Cabin属性
df.head(5)


# In[18]:


EmbarkedDf=pd.DataFrame()#创建pandas中的DataFrame对象
df['Embarked']=df['Embarked'].map(lambda x:x[0])
EmbarkedDf=pd.get_dummies(df['Embarked'],prefix='Embarked')#使用get_dummies进行oone-hot编码，列名前缀是Embarked
EmbarkedDf.head()


# In[19]:


df=pd.concat([df,EmbarkedDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
df.drop('Embarked',axis=1,inplace=True)#删除原来的Embarked属性
df.head(5)


# In[20]:


PclassDf=pd.DataFrame()#创建pandas中的DataFrame对象
PclassDf=pd.get_dummies(df['Pclass'],prefix='Pclass')#使用get_dummies进行oone-hot编码，列名前缀是Pclass
PclassDf.head()


# In[21]:


df=pd.concat([df,PclassDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
df.drop('Pclass',axis=1,inplace=True)#删除原来的Pclass属性
df.head(5)


# In[22]:


sns.countplot(x="SibSp", hue="Survived", data=train)


# In[23]:


sns.countplot(x="Parch", hue="Survived", data=train)


# In[24]:


ParchDf=pd.DataFrame()
ParchDf['Parch_none']=df['Parch'].map(lambda x:1 if x==0 else 0)
ParchDf['Parch_less']=df['Parch'].map(lambda x:1 if 1<=x<=4 else 0)
ParchDf['Parch_many']=df['Parch'].map(lambda x:1 if x>4 else 0)
ParchDf.head()


# In[25]:


df=pd.concat([df,ParchDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
#df.drop('Parch',axis=1,inplace=True)#删除原来的Parch属性
df.head(5)


# In[26]:


SibSpDf=pd.DataFrame()
SibSpDf['SibSp_none']=df['SibSp'].map(lambda x:1 if x==0 else 0)
SibSpDf['SibSp_less']=df['SibSp'].map(lambda x:1 if 1<=x<=4 else 0)
SibSpDf['SibSp_many']=df['SibSp'].map(lambda x:1 if x>4 else 0)
SibSpDf.head()


# In[27]:


df=pd.concat([df,SibSpDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
#df.drop('SibSp',axis=1,inplace=True)#删除原来的属性
df.head(5)


# In[28]:


FamilySize=df['Parch']+df['SibSp']+1
df['FamilySize']=FamilySize
FamilySizeDf=pd.DataFrame()
FamilySizeDf['FamilySize_single']=df['FamilySize'].map(lambda x:1 if x==1 else 0)
FamilySizeDf['FamilySize_less']=df['FamilySize'].map(lambda x:1 if 1<x<=4 else 0)
FamilySizeDf['FamilySize_many']=df['FamilySize'].map(lambda x:1 if x>4 else 0)
FamilySize.head()
df=pd.concat([df,FamilySizeDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
#df.drop('SibSp',axis=1,inplace=True)#删除原来的属性
df.head(5)


# In[29]:


#sns.countplot(x="Fare", hue="Survived", data=train)


# In[30]:


FareDf=pd.DataFrame()
FareDf['Fare_low']=df['Fare'].map(lambda x:1 if x<=2.5 else 0)
FareDf['Fare_high']=df['Fare'].map(lambda x:1 if x>2.5 else 0)
FareDf.head()


# In[31]:


df=pd.concat([df,FareDf],axis=1)#将one-hot产生的虚拟变量加入到数据集中
df.drop('Fare',axis=1,inplace=True)#删除原来的属性
df.head(5)


# In[32]:


def getTitle(name):
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3
NameDf = pd.DataFrame()
Name_mapDict = {

    'Capt':        'Officer',

    'Col':         'Officer',

    'Major':       'Officer',

    'Jonkheer':    'Royalty',

    'Don':         'Royalty',

    'Sir':         'Royalty',

    'Dr':          'Officer',

    'Rev':         'Officer',

    'the Countess':'Royalty',

    'Dona':        'Royalty',

    'Mme':         'Mrs',

    'Mlle':        'Miss',

    'Ms':          'Mrs',

    'Mr':          'Mr',

   'Mrs':         'Mrs',

    'Miss':        'Miss',

    'Master':      'Master',

    'Lady':        'Royalty'
}
NameDf['Title'] = df['Name'].map(getTitle)
NameDf['Title'] = NameDf['Title'].map(Name_mapDict)
NameDf = pd.get_dummies(NameDf['Title'])
NameDf.head()


# In[33]:


df = pd.concat([df,NameDf],axis=1)
df.drop('Name',axis=1,inplace=True)
df.head()


# In[34]:


sex_mapDict={"male":1,"female":0}
df.Sex=df.Sex.map(sex_mapDict)
df.head()


# In[35]:


#df=df.drop('Name',axis=1)
df=df.drop('PassengerId',axis=1)
df=df.drop('Ticket',axis=1)
df.head()


# In[36]:


train.shape


# In[37]:


df_X=pd.concat([PclassDf,
               #ParchDf,
               #SibSpDf,
               FamilySizeDf,
               FareDf,
               df.Sex,
               #cabinDf,
               EmbarkedDf,
               NameDf],
               axis=1
               )
df_X.head()


# In[38]:


sourceRow=891
source_X = df_X.iloc[0:sourceRow]
source_Y = df.loc[:sourceRow-1,'Survived'] 
pred_X=df_X.iloc[sourceRow:]


# In[39]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(source_X,source_Y,train_size=0.8,test_size=0.2)


# In[40]:


print('原始数据集特征',source_X.shape,
      '训练数据集特征',train_X.shape,
      '测试数据集特征',test_X.shape)
print('原始数据集标签',source_Y.shape,
      '训练数据集标签',train_Y.shape,
      '测试数据集标签 ',test_Y.shape)


# In[41]:


source_Y.head()


# In[42]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[43]:


model.fit(train_X,train_Y)


# In[44]:


model.score(test_X,test_Y)


# In[45]:


pred_Y=model.predict(pred_X)
pred_Y=pred_Y.astype(int)
predDf=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_Y})
predDf.head()


# In[46]:


predDf.to_csv('C:\\Users\\YJ\\titanic_pred_LogisticRegression.csv',index=False)


# In[47]:


from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

model = SVC(kernel='rbf', degree=3, gamma=1.7)
model.fit(train_X, train_Y)
print(model)

expected = test_Y
predicted = model.predict(test_X)
print(metrics.classification_report(expected, predicted))       # 输出分类信息


# In[48]:


pred_Y=model.predict(pred_X)
pred_Y=pred_Y.astype(int)
predDf=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_Y})
predDf.head()


# In[49]:


predDf.to_csv('C:\\Users\\YJ\\titanic_pred_svm.csv',index=False)


# In[52]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
#model=GaussianNB()
model = GradientBoostingClassifier()
model.fit(train_X, train_Y)
print(model)
expected = test_Y
predicted = model.predict(test_X)
print(metrics.classification_report(expected, predicted))       # 输出分类信息
pred_Y=model.predict(pred_X)
pred_Y=pred_Y.astype(int)
predDf=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_Y})
predDf.head()


# In[53]:


predDf.to_csv('C:\\Users\\YJ\\titanic_pred_GradientBoosting.csv',index=False)

