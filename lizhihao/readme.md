##	简介

*	使用了两层的模型融合

	Level 1使用了 `RandomForest`, `AdaBoost`, `ExtraTrees`, `GBDT`, `DecisionTree`, `KNN`, `SVM` 一共7个模型

	Level 2使用了XGBoost使用第一层预测的结果作为特征对最终的结果进行预测

*	特征工程

	这里使用了 `Sex`, `Age`, `Pclass`, `Title`, `Parch`, `Name_length`, `fare`, `Title` 作为输入特征

	与特征工程相关的文件放在了文件夹 `feature_analysis` 中

<br>

##	参考资料

*	[Kaggle_Titanic生存预测](https://blog.csdn.net/Koala_Tree/article/details/78725881)
