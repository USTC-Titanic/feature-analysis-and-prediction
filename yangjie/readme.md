##	第一次尝试

*	特征工程

	使用六个特征，分别是 `Pclass`, `Parch`, `SibSp`, `Fare`, `Sex`, `cabin`, `Embarked`

*	模型1

	模型 1 使用了 `LogisticRegression`, 准确率 `74.162%`

*	模型2

	模型 2 使用了 `SVM` , 准确率 `76.55%`

	<br>

##	改进1

*	特征工程

	*	`Parch` 和 `SibSp` 有较大相关性, 两者合并作为一个新特征 `FamilySize`

	*	将 `Age` 划分为四个年龄段

	*	使用特征 `Name`

	综上, 本次改进使用到的特征是 `Pclass, Fare, Sex, FamilySize, Embarked, Name`

*	模型与准确率

	LogisticRegression, SVM, GradientBoosting

	其中 `GradientBoosting` 的预测准确率达到了 `80.382%`

	<br>

##	参考资料

*	[机器学习之泰坦尼克号生存预测](https://www.jianshu.com/p/2d15400671f2)
