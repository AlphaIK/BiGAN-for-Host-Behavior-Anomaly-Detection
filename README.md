BiGAN for Host Behavior Anomaly Detection
==
本代码是基于BiGAN的主机行为异常检测实现，训练BiGAN模型并应用不同loss函数来对主机行为进行异常检测。

---
Dependencies:
--
*	Python 3.6
*	TensorFlow 1.14.0
*	numpy 1.19.5
*	matplotlib 3.2.2

---
Folders:
--
**data文件夹中存放实验数据集：**
*	cmdline_train.npy和cmdline_test.npy分别为模型训练集与测试集；	//代码运行前需要解压文件夹中cmdline.rar至当前文件夹。
*	cmdline.py用于读取数据集。
	
**bigan文件夹中存放BiGAN模型代码：**
*	cmdline_utilities.py为BiGAN网络结构设置；
*	run_cmdline.py中包含BiGAN模型loss函数的设置和模型训练测试过程。
	
**traditional algorithm文件夹中存放其他传统异常检测模型的代码：**
*	IF_cmd.py 孤立森林算法；
*	LOF_cmd.py 局部异常因子算法；
*	Robustcovariance_cmd.py 协方差估计算法。

---
Run:
--
**输入如下命令行即可运行：**

		python main.py bigan cmdline run
	
**positional arguments:**

  		{model}               the name of the model you want to run (bigan)
  		{cmdline}		the name of the dataset you want to run the experiments on
  		{run}                 train the model or evaluate it

**optional arguments:**

 		-h, --help            show this help message and exit
  		--epochs [EPOCHS]     number of epochs you want to train the dataset on
 		--w [W]               weight for the sum of the mapping loss function
  		--rd [RD]             random_seed
  		--loss [{crosse,w,wgp,ls,hinge}]
                        	      the loss function in the model
  		--plot [{n,N,y,Y}]    print dis/enc/gen loss(y/n)

