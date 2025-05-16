# Machine-learning-bridging-battery-data
A code library for reading and preprocessing public battery dataset

## 1.简介（Introduction）
本代码库包含2个电池数据集的读取、预处理、双点特征提取和模型预测的python代码库。该代码旨在方便地访问电池数据，并为电池健康管理领域的分析准备数据。

This codebase contains Python code for reading, preprocessing, two-point feature extraction, and model prediction for 2 battery datasets. The code is designed to provide convenient access to battery data and prepare the data for analysis in the field of battery health management.

## 2.数据集（Datasets）
### 2.1 Dataset 1：[https://doi.org/10.35097/1947]
Battery degradation is critical to the cost-effectiveness and usability of battery-powered products. Aging studies can help to better understand and model degradation and to optimize the operation strategy. Nevertheless, there are only a few comprehensive and freely available aging datasets for these applications.
To our knowledge, the dataset presented in the following is one of the largest published to date. It contains over 3 billion data points from 228 commercial NMC/C+SiO lithium-ion cells aged for more than a year under a wide range of operating conditions. We investigate calendar and cyclic aging and also apply different driving cycles to some of the cells. The dataset includes result data (such as the remaining usable capacity or impedance measured in check-ups) and raw data (i.e., measurement logs with two-second resolution).
The data can be used in a wide range of applications, for example, to model battery degradation, gain insight into lithium plating, optimize operation strategies, or test battery impedance or state estimation algorithms using machine learning or Kalman filtering.

### 2.2 Dataset 2: [https://doi.org/10.7302/7tw1-kc35]
The focus of this research effort is to systematically study the capability of aging diagnostics using cell expansion under variety of aging conditions and states. The data collection campaign is very important to cover various degradation modes to extract the degradation features that will be used to inform, parameterize, and validate the models developed earlier. In the data collection campaign, we are documenting the evolution of the electrical and mechanical characteristics and especially the reversible mechanical measurement. It is important to note that we collect data using newly developed fixtures that enables the simultaneous measurement of mechanical and electrical response under pseudo-constant pressure.

## 3. 代码结构及用法(Code Structure and Usage)
本代码库包含两个文件夹，分别是Dataset1、Dataset2，每个文件夹下都包含.py文件，对应数据集的特征提取方法和机器学习算法进行模型的训练和测试，并采用MAE、MAPE、RMSE和R2评价方法进行结果评估。

This code repository contains two folders, namely Dataset1 and Dataset2. Each folder contains a. py file, which corresponds to the feature extraction methods and machine learning algorithms of the dataset for model training and testing. The results are evaluated using MAE, MAPE, RMSE, and R2 evaluation methods.
