import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from pytorch_lightning import LightningModule
from torch.autograd import Variable
import torch.nn.functional as F
from network.network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import torchvision.utils as vutils

from skimage.metrics import structural_similarity as ssim_func
from torch.optim.lr_scheduler import LambdaLR

from utils.RAdam import RAdam
from utils.utils import denormalize, PSNR, apply_window_level


torch.use_deterministic_algorithms(True, warn_only=True)



class DECT_UNetModule(LightningModule):
    def __init__(self,img_ch,
                        output_ch,
                        criterion,
                        t,
                        decay_rate,
                        decay_epoch,
                        diffonly=False,
                        weight_decay=1e-5,
                        lr=1e-3,
                        beta1=0.9,
                        beta2=0.999,
                        model_type='U_Net'):
        super().__init__()

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.criterion = criterion
        self.time=str(datetime.datetime.now())
        print(f'start time{self.time}')

        # Hyper-parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.model_type = model_type
        self.t=t
        self.decay_rate=decay_rate
        self.decay_epoch=decay_epoch
        self.weight_decay=weight_decay
        self.diffonly=diffonly
        self.save_hyperparameters("model_type",
                                  "img_ch",
                                  "output_ch",
                                  "criterion",
                                  "lr",
                                  "beta1",
                                  "beta2",
                                  "model_type",
                                  "t",
                                  "decay_rate",
                                  "decay_epoch",
                                  "weight_decay")
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_epoch_end_image=[]
        self.test_step_outputs = []
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=1,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=1,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=1,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=1,output_ch=1,t=self.t)


    def configure_optimizers(self):

        weight_params = []
        bias_params = []
        for name, param in self.unet.named_parameters():
            if 'weight' in name:
                weight_params.append(param)
            elif 'bias' in name:
                bias_params.append(param)

        optimizer = optim.Adam([{'params': weight_params, 'weight_decay': self.weight_decay},
                                {'params': bias_params, 'weight_decay': 0}],
                                self.lr,
                               [self.beta1, self.beta2],
                               eps=1e-8)
        lambda_epoch = lambda epoch: self.decay_rate if epoch in self.decay_epoch else 1.0
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_epoch)
        self.lr_schedulers=scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def training_step(self, batch):
        images, GT = batch[0], batch[1]
        images=images.unsqueeze(1)
        GT=GT.unsqueeze(1)
        pred_resi=self.unet.forward(images)
        resi = GT-images
        train_loss = self.criterion(pred_resi,resi)
        L2_reg = torch.tensor(0.).to(images.device)
        L2_lambda = self.weight_decay
        for name, param in self.unet.named_parameters():
            if 'weight' in name:
                L2_reg += torch.norm(param)**2
        L2_loss = L2_lambda * L2_reg
        total_loss= L2_loss + train_loss
        outputs={'train_loss': train_loss ,'L2_loss':L2_loss, 'total_loss':total_loss}
        self.train_step_outputs.append(outputs)
        del images
        del GT
        del pred_resi
        del resi
        return {'loss': total_loss}

    def on_train_epoch_end(self):
        outputs=self.train_step_outputs
        lr=self.lr
        train_loss = [d['train_loss'] for d in outputs]
        L2_loss = [d['L2_loss'] for d in outputs]
        total_loss = [d['total_loss'] for d in outputs]
        current_epoch = self.trainer.current_epoch
        self.log('Train/train_loss', torch.mean(torch.stack(train_loss)), logger=True, on_epoch=True,sync_dist=True)
        print('Train')
        train_loss_mean = torch.mean(torch.stack(train_loss)).item()
        L2_loss_mean = torch.mean(torch.stack(L2_loss)).item()
        total_loss_mean = torch.mean(torch.stack(total_loss)).item()

        epoch=self.trainer.current_epoch
        lr=self.lr
        if epoch in self.decay_epoch:
            self.lr=self.decay_rate*lr
            print('learning rate updated')

        print(f'current_epoch: {current_epoch}, train_loss: {train_loss_mean:.6f}, L2_loss: {L2_loss_mean:.6f},total_loss: {total_loss_mean:.6f},learning_rate:{self.lr:.2e}')
        print()
        self.train_step_outputs.clear()


    def _val_psnr(self, pred_img, y_real):
        pred_img_norm = denormalize(pred_img, self.trainer.datamodule.val_mean, self.trainer.datamodule.val_std)
        y_real_norm = denormalize(y_real, self.trainer.datamodule.val_mean, self.trainer.datamodule.val_std)
        psnrs = []
        for i in range(len(pred_img_norm)):
            gt =  y_real_norm[i]
            psnrs.append(PSNR(gt, pred_img_norm[i],
                              drange=gt.max() - gt.min()))

        return torch.mean(torch.stack(psnrs))

    def _val_ssim(self, pred_img, y_real):
        pred_img_norm = denormalize(pred_img, self.trainer.datamodule.val_mean, self.trainer.datamodule.val_std)
        y_real_norm = denormalize(y_real, self.trainer.datamodule.val_mean, self.trainer.datamodule.val_std)

        ssims = []
        device=y_real_norm.device
        for i in range(pred_img_norm.size(0)):
            pred_i = pred_img_norm[i].squeeze().cpu().numpy()
            real_i = y_real_norm[i].squeeze().cpu().numpy()
            ssims.append(ssim_func(pred_i,real_i))
        ssims = [torch.tensor(s) for s in ssims]
        ssims = torch.mean(torch.stack(ssims))
        ssims = ssims.to(device)
        return ssims

    def validation_step(self, batch,batch_idx):
        current_epoch = self.trainer.current_epoch
        images, GT = batch[0], batch[1]
        patient_id, image_id= batch[2], batch[3]
        images=images.unsqueeze(1)
        GT=GT.unsqueeze(1)
        pred_resi=self.unet.forward(images)
        resi=GT-images
        valid_MSE_loss = self.criterion(pred_resi,resi)
        valid_PSNR_loss= self._val_psnr(pred_resi+images,GT)
        valid_SSIM_loss= self._val_ssim(pred_resi+images,GT)
        outputs={'valid_MSE_loss': valid_MSE_loss,'valid_PSNR_loss': valid_PSNR_loss,'valid_SSIM_loss': valid_SSIM_loss}
        self.validation_step_outputs.append(outputs)
        #if batch_idx<10:
        self.save_valid_images(images, GT, pred_resi, batch_idx, current_epoch, patient_id, image_id)
        del images
        del GT
        del pred_resi
        del patient_id
        del image_id
        return outputs

    def on_validation_epoch_end(self):
        outputs=self.validation_step_outputs
        current_epoch = self.trainer.current_epoch
        val_mse = [o['valid_MSE_loss'] for o in outputs]
        val_psnr = [o['valid_PSNR_loss'] for o in outputs]
        val_ssim = [o['valid_SSIM_loss'] for o in outputs]
        mean_val_mse = torch.mean(torch.stack(val_mse)).item()
        mean_val_psnr = torch.mean(torch.stack(val_psnr)).item()
        mean_val_ssim = torch.mean(torch.stack(val_ssim)).item()
        self.log('Valid/avg_val_mse', mean_val_mse, logger=True, on_epoch=True,sync_dist=True)
        self.log('Valid/avg_val_psnr', mean_val_psnr, logger=True, on_epoch=True,sync_dist=True)
        self.log('Valid/avg_val_ssim', mean_val_ssim, logger=True, on_epoch=True,sync_dist=True)
        print('Valid')
        print(f'current_epoch: {current_epoch}, mean_val_mse: {mean_val_mse:6f},mean_val_psnr:{mean_val_psnr:6f},mean_val_ssim:{mean_val_ssim:6f}')
        self.validation_epoch_end_image.clear()
        self.validation_step_outputs.clear()


    def save_valid_images(self, images, GT, pred_resi, batch_idx, current_epoch, patient_id, image_id):
        save_path = '/data_new3/username/DualEnergyCTSynthesis/output/valid_image'+str(self.time)+'/'
        os.makedirs(save_path, exist_ok=True)

        images1 = apply_window_level(images - 1024, 1500, 300)
        GT1 = apply_window_level(GT - 1024, 1500, 300)
        pred_img1 = apply_window_level(pred_resi + images - 1024, 1500, 300)
        diff = GT1 - pred_img1

        images1 = images1.cpu()
        GT1 = GT1.cpu()
        pred_img1 = pred_img1.cpu()
        diff = diff.cpu()
        if current_epoch==0:
            vutils.save_image(GT1, os.path.join(save_path, f'GT_Epoch_{current_epoch}_batch_{batch_idx}_p_{patient_id}_i_{patient_id}.png'), normalize=True)
        vutils.save_image(pred_img1, os.path.join(save_path, f'predict_Epoch_{current_epoch}_batch_{batch_idx}_p_{patient_id}_i_{patient_id}.png'), normalize=True)
        vutils.save_image(diff, os.path.join(save_path, f'diff_Epoch_{current_epoch}_batch_{batch_idx}_p_{patient_id}_i_{patient_id}.png'), normalize=True)
        del images1
        del GT1
        del pred_img1
        del diff
        del images
        del GT
        del pred_resi


    def test_step(self, batch, batch_idx):
        images, GT = batch[0], batch[1]
        patient_id, image_id= batch[2], batch[3]
        images = images.unsqueeze(1)
        GT = GT.unsqueeze(1)
        pred_resi = self.unet.forward(images)
        resi = GT - images
        test_MSE_loss = self.criterion(pred_resi, resi)  
        test_PSNR_loss= self._val_psnr(pred_resi+images,GT)
        test_SSIM_loss= self._val_ssim(pred_resi+images,GT)
        self.save_test_images(images, GT, pred_resi, batch_idx, patient_id, image_id)
        outputs={"test_loss": test_MSE_loss,
                 "PSNR_loss":test_PSNR_loss,
                 "SSIM_loss":test_SSIM_loss}
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        outputs=self.test_step_outputs
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_PSNR_loss = torch.stack([x['PSNR_loss'] for x in outputs]).mean()
        avg_SSIM_loss = torch.stack([x['SSIM_loss'] for x in outputs]).mean()


        print(f"Test completed: Average MSE loss: {avg_test_loss.item()}")
        print(f"Test completed: Average PSNR loss: {avg_PSNR_loss.item()}")
        print(f"Test completed: Average SSIM loss: {avg_SSIM_loss.item()}")


    def save_test_images(self, images, GT, pred_resi, batch_idx, patient_id, image_id):
        save_path = '/data_new3/username/DualEnergyCTSynthesis/output/test_image'
        os.makedirs(save_path, exist_ok=True)

        images1 = apply_window_level(images - 1024, 1500, 300)
        GT1 = apply_window_level(GT - 1024, 1500, 300)
        pred_img1 = apply_window_level(pred_resi + images - 1024, 1500, 300)
        diff = GT1 - pred_img1

        images1 = images1.cpu()
        GT1 = GT1.cpu()
        pred_img1 = pred_img1.cpu()
        diff = diff.cpu()

        vutils.save_image(GT1, os.path.join(save_path, f'batch_{batch_idx}_{self.time}_patient_id_{patient_id}_image_id_{patient_id}_GT.png'), normalize=True)
        vutils.save_image(pred_img1, os.path.join(save_path, f'batch_{batch_idx}_{self.time}_patient_id_{patient_id}_image_id_{patient_id}_predict.png'), normalize=True)
        vutils.save_image(diff, os.path.join(save_path, f'batch_{batch_idx}_{self.time}_patient_id_{patient_id}_image_id_{patient_id}_diff.png'), normalize=True)
