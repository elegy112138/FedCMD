import re
from collections import defaultdict

# 初始化一个字典来统计每个层名称的出现次数
layer_counts = defaultdict(int)

# 打开并读取日志文件的内容
with open('kl_log_FedSpeR_dyn.txt', 'r') as file:
    content = file.read()
    # 使用正则表达式查找所有层的名称
    matches = re.findall(r'Layer with minimum contribution to output y: (\w+)', content)
    for match in matches:
        layer_counts[match] += 1

# 打印统计结果
for layer, count in layer_counts.items():
    print(f"{layer}: {count} times")
