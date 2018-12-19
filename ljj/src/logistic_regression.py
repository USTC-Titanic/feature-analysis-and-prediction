'''
使用逻辑回归进行预测
特征工程参加群文档[]
这里使用了 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch' 作为输入特征
模型构造的参考资料
https://www.kaggle.com/c/titanic
https://blog.csdn.net/chenhaifeng2016/article/details/73136084
注1: 参考模型中的 .as_matrix() 即将被 pd 弃用, 新的用法是 .values
注2: 参考模型中使用了 6 个特征, 而由我们之前的特征工程, 只用5个特征即可, 根据性能进行调试
'''

import numpy as np
import pandas as pd
import tensorflow as tf

def feature_engineering_test_data():
	# 读取测试数据集
	data_test = pd.read_csv('test.csv')
	# 获取乘客ID，方便构造最后要提交的数据
	PassengerId = data_test['PassengerId']
	# 将性别从字符串类型转换为0或1数值型数据
	data_test['Sex'] = data_test['Sex'].apply(lambda s: 1 if s == 'male' else 0)
	
	# 计算年龄平均值
	age_mean = data_test.Age.mean()
	# 年龄的缺失值用平均值进行填充
	data_test.Age = data_test.Age.fillna(age_mean)

	# 其他的缺失值填0
	data_test = data_test.fillna(0)

	# 选取特征
	dataset_X = data_test[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch']].values
	return PassengerId, dataset_X

def feature_engineering_train_data():
	# 读训练数据
	data_train = pd.read_csv('train.csv')
	# 将性别从字符串类型转换为0或1数值型数据
	data_train['Sex'] = data_train['Sex'].apply(lambda s: 1 if s == 'male' else 0)

	age_mean = data_train.Age.mean()
	# 年龄的缺失值用平均值进行填充
	data_train.Age = data_train.Age.fillna(age_mean)

	# 其他缺失值填0
	data_train = data_train.fillna(0)

	# 选取特征
	dataset_X = data_train[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch']].values

	# 建立标签
	data_train['Deceased'] = data_train['Survived'].apply(lambda x: 1 - x)
	dataset_Y = data_train[['Deceased', 'Survived']].values

	return dataset_X, dataset_Y

def train():
	# 定义占位符
	x = tf.placeholder(tf.float32, shape=[None, 5])
	y = tf.placeholder(tf.float32, shape=[None, 2])

	# 使用逻辑回归模型
	# y = σ(wx + b)
	weights = tf.Variable(tf.random_normal([5, 2]), name='weights')
	bias = tf.Variable(tf.zeros([2]), name='bias')
	y_pred = tf.nn.softmax(tf.matmul(x, weights) + bias)

	# 定义交叉熵
	cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
	#定义损失函数
	cost = tf.reduce_mean(cross_entropy)

	# 使用梯度下降优化算法最小化损失函数
	lr = 0.001
	train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

	with tf.Session() as sess:
		print('start training')
		# 变量初始化
		tf.global_variables_initializer().run()

		dataset_X, dataset_Y = feature_engineering_train_data()
		# 训练模型, 50 个 epoch
		for epoch in range(50):
			total_loss = 0
			for i in range(len(dataset_X)):
				# prepare feed data and run
				feed_dict = {x: [dataset_X[i]], y: [dataset_Y[i]]}
				var, loss = sess.run([train_op, cost], feed_dict=feed_dict)
				total_loss += loss
			# display loss per epoch
			print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
		print("Train Complete")

		PassengerId, X_test = feature_engineering_test_data()
		# 测试模型
		predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)
		
		# 保存结果
		submission = pd.DataFrame({
			"PassengerId": PassengerId,
			"Survived": predictions
		})
		submission.to_csv("titanic-submission.csv", index=False)
		print('result saved')

if __name__ == '__main__':
	train()
