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
#batchsize=16时GPU占用为20000M,更高会OOM
dm = NanjingPLA_DECT(datatype='train',train_root_dir='/data_new3/username/DualEnergyCTSynthesis/dataset/train',valid_root_dir='/data_new3/username/DualEnergyCTSynthesis/dataset/valid',
                     test_root_dir=None, batch_size=1, gt_shape=512)


print('data is loaded')


#1~2卡效率最高,4卡更慢
trainer = Trainer(max_epochs=200,
                  devices=1,
                  default_root_dir='/data_new3/username/DualEnergyCTSynthesis/output',
                  callbacks=[ModelCheckpoint(
                                            dirpath=None,
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            mode='min'
                                        )],
                  deterministic=True,
                  num_sanity_val_steps=0,
                  #pin_memory=True,
                  precision='16-mixed',
                  profiler=None,
                  #limit_train_batches=0.2,
                  #limit_val_batches=0.5,
                  enable_progress_bar=True,
                  fast_dev_run=False,
                  log_every_n_steps=5,
                  check_val_every_n_epoch=1,
                  strategy='ddp')

#config.model_type in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:


model = DECT_UNetModule(lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        model_type='U_Net',
        img_ch=1,
        output_ch=1,
        criterion=torch.nn.MSELoss(),
        t=3,
        weight_decay=0,
        decay_rate=0.5,
        decay_epoch=(2,8,16,32,64,100,200))
init_weights(model, init_type='kaiming')

trainer.fit(model, datamodule=dm)
    #elif config.mode == 'test':
    #    model.test()
