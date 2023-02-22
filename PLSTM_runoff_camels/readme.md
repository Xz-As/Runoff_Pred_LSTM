基准 数据：00000001_data.txt
计算后的其他水文数据：00000001_help.txt
其他水文数据的计算程序：param_extend.py
模型运行：Main.py（28个epoch，nse达到0.646左右）


运行程序时，若要采用其他数据，请先将数据覆盖到00000001_data.txt，形式为：
N	P	E	T	Q


运行说明：
`basin`为存储其他数据的txt的名称前8位数字
名称格式为：`basin`_data.txt
若要使用其他数据，请事先执行：
"""
python param_extend.py -b basin
"""

若要运行带物理层的P-LSTM，请执行以下指令：
"""
python Main.py
"""

若要运行不带物理层的LSTM，请执行以下指令：
"""
python Main.py -n
"""
或：
"""
python Main.py --nn
"""

若预测其他数据，请执行
"""
python Main.py -i basin
"""
以上选项可组合
若要查看帮助，请执行
"""
python Main.py -h
"""
"""
python Main.py --help
"""

若输入其他选项，将运行P-LSTM


输出说明：
首先得到一张图，为预测值，如果是自带数据数据，还将包含观测值与nse值
然后将所有预测值与观测值存入basin_output.txt，存储格式为制表符间隔的csv格式
