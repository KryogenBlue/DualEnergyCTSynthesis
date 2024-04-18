import os
import numpy as np
import io
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 函数来绘制损失曲线
def plot_loss_curves(loss_dict, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for loss_name, loss_values in loss_dict.items():
        plt.figure()
        plt.plot(loss_values, label=loss_name)
        plt.title(loss_name)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        loss_name=str(loss_name).replace('/','')
        plt.savefig(os.path.join(save_dir, f"{loss_name}.png"))
        plt.close()

# 函数来保存图像
def save_images(event_acc, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_tags = event_acc.Tags()['images']
    for tag in image_tags:
        tag_dir = os.path.join(save_dir, tag)
        if not os.path.exists(tag_dir):
            os.makedirs(tag_dir)
        images = event_acc.Images(tag)
        tag=str(tag).replace('/','')
        for i, image in enumerate(images):
            img = plt.imread(io.BytesIO(image.encoded_image_string))
            plt.imsave(os.path.join(tag_dir, f"{tag}_{i}.png"), img)#,#cmap='gray', vmin=-500, vmax=500)

# 路径到TensorBoard日志文件
event_file_path = '/data_new3/username/DualEnergyCTSynthesis/output/lightning_logs/version_85/events.out.tfevents.1700913977.amax.3575726.0'

# 创建一个事件累加器实例来收集事件数据
event_acc = EventAccumulator(event_file_path)
event_acc.Reload()  # 加载事件文件

# 提取损失数据
loss_scalars = {}
for tag in event_acc.Tags()['scalars']:
    events = event_acc.Scalars(tag)
    loss_scalars[tag] = [event.value for event in events]

# 绘制并保存损失曲线
loss_save_dir = '/data_new3/username/DualEnergyCTSynthesis/output/'+'version_85'+'/loss_curves'
plot_loss_curves(loss_scalars, loss_save_dir)

# 提取并保存图像
image_save_dir = '/data_new3/username/DualEnergyCTSynthesis/output/'+'version_85'+'/images'
save_images(event_acc, image_save_dir)
