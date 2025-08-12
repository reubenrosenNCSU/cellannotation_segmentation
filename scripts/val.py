from ultralytics import YOLO
import csv
from pathlib import Path
# 加载你训练目录下的 last.pt 模型文件
model_path = "snapshots/best.pt" # 根据你的实际路径修改
model = YOLO(model_path)
# 指定数据集配置文件路径
dataset_config_path = "/home/greenbaumgpu/Fengyi/YOLO/datasets/CD3_2_2/images/data.yaml" # 根据你的实际路径修改

print(f"Evaluating model {model_path} on test dataset specified in {dataset_config_path}...")
results = model.val(data=dataset_config_path, split='test') # 确保这里的 'test' 或 'val' 与你的YAML文件配置一致

p = results.box.p     # 平均 Precision
r = results.box.r     # 平均 Recall
mAP50 = results.box.map50
print(f"\nmAP@0.5 on the test dataset: {mAP50}")
mAP = results.box.map
print(f"mAP@0.5:0.95 on the test dataset: {mAP}")

# results.save_dir 就是 model.val() 自动创建的那个目录的路径
save_directory = Path(results.save_dir)
# 定义你想要保存的 CSV 文件名
csv_filename = "metrics_summary.csv" # 或者你喜欢的其他名字
# 构建完整的 CSV 文件路径：目录路径 + 文件名
csv_filepath = save_directory / csv_filename # <-- 就像这样组合路径
# ... 后面的代码就是将数据写入到 csv_filepath 这个路径指定的文件中 ...
with open(csv_filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Precision', p])
    writer.writerow(['Recall', r])
    writer.writerow(['mAP@0.5', mAP50])
    writer.writerow(['mAP@0.5:0.95', mAP])
# 打印保存路径，方便你找到文件
print(f"Metrics have been saved to {csv_filepath}")