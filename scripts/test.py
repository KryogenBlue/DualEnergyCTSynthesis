import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#export CUDA_VISIBLE_DEVICES=0,1,3,2
import torch

import numpy as np

from modules.DECT_UNetModule import DECT_UNetModule
from network.network import init_weights

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from build_dataset.PLADataModule import NanjingPLA_DECT

seed_everything(42)
torch.use_deterministic_algorithms(True, warn_only=True)

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.benchmark=False

torch.backends.cudnn.deterministic=True

print('gpu:',torch.cuda.is_available())

dm = NanjingPLA_DECT(datatype='test',train_root_dir='/data_new3/username/DualEnergyCTSynthesis/dataset/train',valid_root_dir='/data_new3/username/DualEnergyCTSynthesis/dataset/valid',
                     test_root_dir='/data_new3/username/DualEnergyCTSynthesis/dataset/test', batch_size=1, gt_shape=512)


print('data is loaded')


trainer = Trainer(devices=1,
                  default_root_dir='/data_new3/username/DualEnergyCTSynthesis/output/test',
                  callbacks=[ModelCheckpoint(
                                            dirpath=None,
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            mode='min'
                                        )],
                  deterministic=True,
                  num_sanity_val_steps=0,
                  precision='16-mixed',
                  enable_progress_bar=True)

#config.model_type in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:

model_path = '/data_new3/username/DualEnergyCTSynthesis/output/lightning_logs/version_17/checkpoints/epoch=199-step=62000.ckpt'
model = DECT_UNetModule.load_from_checkpoint(model_path)
trainer.test(model, datamodule=dm)