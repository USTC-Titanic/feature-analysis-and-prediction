#!/usr/bin/env python
# coding: utf-8

# In[62]:


# 数据分析和探索
import pandas as pd
import numpy as np
import random as rnd
import tensorflow as tf

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


# 获取数据，训练集train_df，测试集test_df，合并集合combine（便于对特征进行处理时统一处理：for df in combine:）
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]


# In[64]:


train_df.head()


# In[65]:


# 查看各特征非空样本量及字段类型
train_df.info()
print("_"*40)
test_df.info()


# In[66]:


# 查看数值类（int，float）特征的数据分布情况
train_df.describe()


# In[67]:


# 查看非数值类（object类型）特征的数据分布情况
train_df.describe(include=["O"])


# In[68]:


train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 富人和中等阶层有更高的生还率，底层生还率低


# In[69]:


train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 性别和是否生还强相关，女性用户的生还率明显高于男性


# In[70]:


train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 有0到2个兄弟姐妹或配偶的生还几率会高于有更多的


# In[71]:


train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending=False)
# 相关


# In[72]:


g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Age",bins=20)
# 婴幼儿的生存几率更大


# In[73]:



# Fare
g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Fare",bins=10)
# 票价最便宜的幸存几率低


# In[74]:


grid = sns.FacetGrid(train_df,row="Survived",col="Sex",aspect=1.6)
grid.map(plt.hist,"Age",alpha=.5,bins=20)
grid.add_legend()
# 女性的幸存率更高，各年龄段均高于50%
# 男性中只有婴幼儿幸存率高于50%，年龄最大的男性（近80岁）幸存


# In[75]:


grid1 = sns.FacetGrid(train_df,col="Embarked")
grid1.map(sns.pointplot,"Pclass","Survived","Sex",palette = "deep")
#


# In[76]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[77]:


# Some features of my own that I have added in
# Gives the length of the name
train_df['NameLength'] = train_df['Name'].apply(len)
test_df['NameLength'] = test_df['Name'].apply(len)


# In[78]:


# Feature that tells whether a passenger had a cabin on the Titanic
train_df['HasCabin'] = train_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_df['HasCabin'] = test_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


# In[79]:


# 剔除Ticket（人为判断无关联）和Cabin（有效数据太少）两个特征
train_df = train_df.drop(["Ticket","Cabin"],axis=1)
test_df = test_df.drop(["Ticket","Cabin"],axis=1)
combine = [train_df,test_df]
print(train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)


# In[80]:


# 根据姓名创建称号特征，会包含性别和阶层信息
# dataset.Name.str.extract(' ([A-Za-z]+)\.' -> 把空格开头.结尾的字符串抽取出来
# 和性别匹配，看各类称号分别属于男or女，方便后续归类

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex']).sort_values(by=["male","female"],ascending=False)


# In[81]:


# 把称号归类为Mr,Miss,Mrs,Master,Rare_Male,Rare_Female(按男性和女性区分了Rare)
for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess', 'Dona'],"Rare_Female")
    dataset["Title"] = dataset["Title"].replace(['Capt', 'Col','Don','Dr','Major',
                                                 'Rev','Sir','Jonkheer',],"Rare_Male")
    dataset["Title"] = dataset["Title"].replace('Mlle', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')


# In[82]:


# 按Title汇总计算Survived均值，查看相关性
train_df[["Title","Survived"]].groupby(["Title"],as_index=False).mean()


# In[83]:


# title特征映射为数值
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare_Female":5,"Rare_Male":6}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)
    # 为了避免有空数据的常规操作
train_df.head()


# In[84]:


# Name字段可以剔除了
# 训练集的PassengerId字段仅为自增字段，与预测无关，可剔除
train_df = train_df.drop(["Name","PassengerId"],axis=1)
test_df = test_df.drop(["Name"],axis=1)


# In[85]:


# 每次删除特征时都要重新combine
combine = [train_df,test_df]
combine[0].shape,combine[1].shape


# In[86]:


# sex特征映射为数值
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"female":1,"male":0}).astype(int)
    # 后面加astype(int)是为了避免处理为布尔型？
train_df.head()


# In[87]:


# 对Age字段的空值进行预测补充
# 取相同Pclass和Title的年龄中位数进行补充（Demo为Pclass和Sex）

grid = sns.FacetGrid(train_df,col="Pclass",row="Title")
grid.map(plt.hist,"Age",bins=20)


# In[88]:


guess_ages = np.zeros((6,3))
guess_ages


# In[89]:


# 给age年龄字段的空值填充估值
# 使用相同Pclass和Title的Age中位数来替代（对于中位数为空的组合，使用Title整体的中位数来替代）


for dataset in combine:
    # 取6种组合的中位数
    for i in range(0, 6):
        
        for j in range(0, 3):
            guess_title_df = dataset[dataset["Title"]==i+1]["Age"].dropna()
            
            guess_df = dataset[(dataset['Title'] == i+1) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median() if ~np.isnan(guess_df.median()) else guess_title_df.median()
            #print(i,j,guess_df.median(),guess_title_df.median(),age_guess)
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    # 给满足6中情况的Age字段赋值
    for i in range(0, 6):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Title == i+1) & (dataset.Pclass == j+1),
                        'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[90]:


#创建是否儿童特征
for dataset in combine:
    dataset.loc[dataset["Age"] > 12,"IsChildren"] = 0
    dataset.loc[dataset["Age"] <= 12,"IsChildren"] = 1
train_df.head()


# In[91]:


# 创建年龄区间特征
# pd.cut是按值的大小均匀切分，每组值区间大小相同，但样本数可能不一致
# pd.qcut是按照样本在值上的分布频率切分，每组样本数相同
train_df["AgeBand"] = pd.qcut(train_df["Age"],8)
train_df[["AgeBand","Survived"]].groupby(["AgeBand"],as_index = False).mean().sort_values(by="AgeBand",ascending=True)


# In[92]:


# 把年龄按区间标准化为0到4
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 17, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 21), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 21) & (dataset['Age'] <= 25), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 26), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 31), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 36.5), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 36.5) & (dataset['Age'] <= 45), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 7
train_df.head()


# In[93]:


# 移除AgeBand特征
train_df = train_df.drop(["AgeBand"],axis=1)
combine = [train_df,test_df]
train_df.head()


# In[94]:


# 创建家庭规模FamilySize组合特征
for dataset in combine:
    dataset["FamilySize"] = dataset["Parch"] + dataset["SibSp"] + 1
train_df[["FamilySize","Survived"]].groupby(["FamilySize"],as_index = False).mean().sort_values(by="FamilySize",ascending=True)


# In[95]:


# 创建是否独自一人IsAlone特征
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1,"IsAlone"] = 1
train_df[["IsAlone","Survived"]].groupby(["IsAlone"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[96]:


# 移除Parch,Sibsp,FamilySize（暂且保留试试）
# 给字段赋值可以在combine中循环操作，删除字段不可以，需要对指定的df进行操作
train_df = train_df.drop(["Parch","SibSp"],axis=1)
test_df = test_df.drop(["Parch","SibSp"],axis=1)
combine = [train_df,test_df]
train_df.head()


# In[97]:


# 给Embarked补充空值
# 获取上船最多的港口
freq_port = train_df["Embarked"].dropna().mode()[0]
freq_port


# In[98]:


for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
train_df[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[99]:


# 把Embarked数字化
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map({"S":0,"C":1,"Q":2}).astype(int)
train_df.head()


# In[100]:


# 去掉Embarked试试。。
#train_df = train_df.drop(["Embarked"],axis=1)
#test_df = test_df.drop(["Embarked"],axis=1)
#combine=[train_df,test_df]
#train_df.head()


# In[101]:


# 给测试集中的Fare填充空值，使用中位数
test_df["Fare"].fillna(test_df["Fare"].dropna().median(),inplace=True)
test_df.info()


# In[102]:


# 创建FareBand区间特征
train_df["FareBand"] = pd.qcut(train_df["Fare"],4)
train_df[["FareBand","Survived"]].groupby(["FareBand"],as_index=False).mean().sort_values(by="FareBand",ascending=True)


# In[103]:


# 根据FareBand将Fare特征转换为序数值
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[104]:


test_df.head(10)


# In[105]:


# 用seaborn的heatmap对特征之间的相关性进行可视化
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[106]:


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test_ = test_df.drop("PassengerId",axis=1).copy()
X_train.shape,Y_train.shape,X_test.shape


# In[107]:


#训练集：dataset_X，dataset_Y     
dataset_X = X_train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','NameLength','HasCabin','Title','IsChildren','FamilySize','IsAlone']].values
train_df['Deceased'] = train_df['Survived'].apply(lambda x: 1 - x)
dataset_Y=train_df[['Deceased', 'Survived']].values
#测试集：X_test
X_test=X_test_[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked','NameLength','HasCabin','Title','IsChildren','FamilySize','IsAlone']].values
PassengerId=test_df['PassengerId']


# In[108]:


x = tf.placeholder(tf.float32, shape=[None, 11])
y = tf.placeholder(tf.float32, shape=[None,2])
# 使用逻辑回归模型
#y = σ(wx + b)
weights = tf.Variable(tf.random_normal([11, 2]), name='weights')
bias = tf.Variable(tf.zeros([2]), name='bias')
y_pred = tf.nn.softmax(tf.matmul(x, weights) + bias)
# 定义交叉熵
cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
#定义损失函数
cost = tf.reduce_mean(cross_entropy)
# 使用梯度下降优化算法最小化损失函数
lr = 0.001
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)


# In[109]:


with tf.Session() as sess:
    print('start training')
    tf.global_variables_initializer().run()
    for epoch in range(70):
            total_loss = 0
            for i in range(len(dataset_X)):
            # prepare feed data and run
                feed_dict = {x: [dataset_X[i]], y: [dataset_Y[i]]}
                var, loss = sess.run([train_op, cost], feed_dict=feed_dict)
                total_loss += loss
            # display loss per epoch
            print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    print("Train Complete")

    # 测试模型
    predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)
    # 保存结果
    submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": predictions
    })
    submission.to_csv("titanic-submission.csv", index=False)
    print('result saved')

