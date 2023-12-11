# 主要环境依赖
```
pyhton 3.10.11
torch 2.1.0
torch-geometric 2.3.1
```
# 相关文件下载
可以下载[原始数据集](https://www.biendata.xyz/competition/smpecisa2019/)进行本地处理，也可以直接下载[预处理后的数据集](https://doi.org/10.57760/sciencedb.14119)，置于`datasets/new`路径下。  
可以下载[完整conceptnet](https://github.com/commonsense/conceptnet5/wiki/Downloads)进行本地处理，也可以联系作者<lijiawei13145@outlook.com>获取本地处理后的知识库，置于`datasets/inter-file`路径下。  
下载[预训练的中文词嵌入模型](https://github.com/Embedding/Chinese-Word-Vectors)，置于`model/pre_train`路径下。
# 运行
```shell
python run.py --slide 3 --in_channels 300 --hidden_channels 768 --num_heads 3 --layers 3 --dropout 0.4 --lr 6e-6 --weight_decay 5e-5 --batch_size 16 
```



