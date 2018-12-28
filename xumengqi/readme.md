##	初次尝试

*	使用逻辑回归进行预测

*	特征工程

	与特征工程相关的文件放在了文件夹 `feature_analysis` 中

	这里使用了 `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `NameLength`, `HasCabin`, `Title`, `IsChildren`, `FamilySize`, `IsAlone` 作为输入特征

*	代码放在了 `src` 文件夹下

*	结果放在了 `result` 文件夹下

	<br>

##	改进1

*	特征工程

	*	从 `name` 中提取 `title`

	*	将 `sib` 和 `parch` 相加得到 `FamilyLabel`

	*	将 `ticket` 分组

	*	对遇难组的乘客信息做惩罚性修改

	*	进行 `one-hot` 编码

	综上, 本次改进后使用到的特征是 `Survived, Pclass, Sex, Age, Fare, Embarked, Title, FamilyLabel, Deck, TicketGroup`

*	模型与准确率

	使用逻辑回归模型, 预测准确率为 `80.861%`

##	参考资料

*	[Titanic from Kaggle](https://www.kaggle.com/c/titanic)

*	[Kaggle实战——泰坦尼克号生还预测](https://mastervan.github.io/2017/04/25/kaggle%E5%AE%9E%E6%88%98%E2%80%94%E2%80%94%E6%B3%B0%E5%9D%A6%E5%B0%BC%E5%85%8B%E5%8F%B7%E7%94%9F%E8%BF%98%E9%A2%84%E6%B5%8B/)
