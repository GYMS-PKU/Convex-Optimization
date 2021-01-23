MATLAB 版本: 9.7.0.1190202 (R2019b)
CVX: Version 2.2, Build 1148 (62bfcca)  
MOSEK version    : 9.2.29
Gurobi Optimizer ：version 9.1.0 build v9.1.0rc0 (win64)

四个算法分别封装在gl_cvx_mosek.m、gl_cvx_gurobi.m、gl_SGD_primal.m、gl_GD_primal.m中
直接运行main.m中第一部分的代码即可选定随机种子进行实验
其余部分的代码完成制图
另外实现了带BB步长的光滑化算法，封装在gl_gdbb.m中，仅供报告分析使用

绘图函数没有完全参照课程提供的plot_result

norms函数是cvx工具箱自带的计算范数的函数，本程序中直接调用

多次实验的随机种子已经在main.m中指定，直接运行可复现结果