import os
import numpy as np
import io
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_combined_loss_curves(event_file_path_1, event_file_path_2, save_dir):
    # 加载两组日志数据
    event_acc_1 = EventAccumulator(event_file_path_1)
    event_acc_1.Reload()
    event_acc_2 = EventAccumulator(event_file_path_2)
    event_acc_2.Reload()

    # 提取两组损失数据
    loss_scalars_1 = {tag: [event.value for event in event_acc_1.Scalars(tag)] for tag in event_acc_1.Tags()['scalars']}
    loss_scalars_2 = {tag: [event.value for event in event_acc_2.Scalars(tag)] for tag in event_acc_2.Tags()['scalars']}

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制并保存组合损失曲线
    for tag in set(loss_scalars_1).union(set(loss_scalars_2)):
        plt.figure()
        if tag in loss_scalars_1:
            plt.plot(loss_scalars_1[tag], label=f'{tag} (Group 1)')
        if tag in loss_scalars_2:
            plt.plot(loss_scalars_2[tag], label=f'{tag} (Group 2)')
        plt.title(tag)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        tag=str(tag).replace('/','')
        plt.savefig(os.path.join(save_dir, f"{tag}.png"))
        plt.close()

# 示例使用
event_file_path_1 = '/data_new3/username/DualEnergyCTSynthesis/output/lightning_logs/version_87/events.out.tfevents.1700931690.amax.419377.0'
event_file_path_2 = '/data_new3/username/DualEnergyCTSynthesis/output/lightning_logs/version_88/events.out.tfevents.1700937164.amax.917435.0'
loss_save_dir = '/data_new3/username/DualEnergyCTSynthesis/output/'+'version_87&88'
plot_combined_loss_curves(event_file_path_1, event_file_path_2, loss_save_dir)