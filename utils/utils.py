import numpy as np
import scipy
import torch
import torch.fft

import torch.nn.functional as F


def log_amplitudes(amp):
    """
    Compute log-amplitudes of a Fourier spectrum.

    :param fft: 
    :return: log-amplitudes
    """
    amp[amp == 0] = 1.
    return torch.log(amp)

def downsample_fourier_torch(fourier_coeffs, new_shape):
    """
    Downsample Fourier coefficients to a new shape (Torch version).
    
    Parameters:
        fourier_coeffs (torch.Tensor): Original Fourier coefficients.
        new_shape (tuple): Shape of the downsampled Fourier coefficients.
        
    Returns:
        torch.Tensor: Downsampled Fourier coefficients.
    """
    return fourier_coeffs[:, :new_shape[0], :new_shape[1]].float()

def downsample_image(image_tensor, target_shape):
    """
    Downsample a 512x512 image tensor to a target shape using bilinear interpolation.

    :param image_tensor: A PyTorch tensor of shape [1, C, 512, 512] where C is the number of channels
    :param target_shape: A tuple or list of the target height and width (H, W)
    :return: A PyTorch tensor of the image downsampled to target_shape
    """
    # 假设 image_tensor 最初是 [512, 512], 我们需要它是 [1, C, 512, 512]
    if len(image_tensor.shape) == 2:  # 如果没有通道维度
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加批处理和通道维度
    elif len(image_tensor.shape) == 3:  # 如果已经有了通道维度
        image_tensor = image_tensor.unsqueeze(0)  # 只添加批处理维度

    # 使用双线性插值进行降采样
    downsampled_tensor = F.interpolate(image_tensor, size=target_shape, mode='bilinear', align_corners=False)

    return downsampled_tensor.squeeze(0)


def interpolate_fourier_torch(downsampled_coeffs, original_shape):
    """
    Interpolate downsampled Fourier coefficients back to the original shape (Torch version).
    
    Parameters:
        downsampled_coeffs (torch.Tensor): Downsampled Fourier coefficients.
        original_shape (tuple): Original shape of the Fourier coefficients.
        
    Returns:
        torch.Tensor: Interpolated Fourier coefficients.
    """
    return F.interpolate(downsampled_coeffs.unsqueeze(1), size=original_shape, mode='bilinear', align_corners=True).squeeze(1)

def normalize_FC(rfft, amp_min, amp_max):
    """
    Convert Fourier coefficients of rFFT into normalized amplitudes and phases.
    
    :param rfft: 
    :param amp_min: 
    :param amp_max: 
    :return: 
    """
    amp = rfft.abs()
    phi = rfft.angle()

    amp = normalize_amp(amp, amp_min=amp_min, amp_max=amp_max)
    phi = normalize_phi(phi)

    return amp, phi


'''def denormalize_FC(fc, amp_min, amp_max):
    """
    Convert normalized amplitudes and phases `x` into Fourier coefficients.
    
    :param fc: 
    :param amp_min: 
    :param amp_max: 
    :return: 
    """
    amp = denormalize_amp(fc[..., 0], amp_min=amp_min, amp_max=amp_max)
    phi = denormalize_phi(fc[..., 1])
    return torch.complex(amp * torch.cos(phi), amp * torch.sin(phi))


def convert2DFT(x, amp_min, amp_max, dst_flatten_order, img_shape=27):
    """
    Convert normalized amplitudes and phases `x` into discrete Fourier transform.
    
    :param x: 
    :param amp_min: 
    :param amp_max: 
    :param dst_flatten_order: flattening order of `x`
    :param img_shape: real-space image shape
    :return: 
    """
    x = denormalize_FC(x, amp_min, amp_max)

    dft = torch.ones(x.shape[0], img_shape * (img_shape // 2 + 1), dtype=x.dtype, device=x.device)
    dft[:, :x.shape[1]] = x

    dft[:, dst_flatten_order] = torch.flatten(dft.clone(), start_dim=1)
    return dft.reshape(-1, img_shape, img_shape // 2 + 1)'''
def denormalize_FC(fc, amp_min, amp_max):
    """
    Convert normalized amplitudes and phases `x` into Fourier coefficients.
    
    :param fc: 
    :param amp_min: 
    :param amp_max: 
    :return: real and imaginary parts as separate tensors
    """
    amp = denormalize_amp(fc[..., 0], amp_min=amp_min, amp_max=amp_max)
    phi = denormalize_phi(fc[..., 1])
    real_part = amp * torch.cos(phi)
    imag_part = amp * torch.sin(phi)
    return real_part, imag_part

def convert2DFT(x, amp_min, amp_max, dst_flatten_order, img_shape=27):
    """
    Convert normalized amplitudes and phases `x` into discrete Fourier transform.
    
    :param x: 
    :param amp_min: 
    :param amp_max: 
    :param dst_flatten_order: flattening order of `x`
    :param img_shape: real-space image shape
    :return: real and imaginary parts as separate tensors
    """
    real_part, imag_part = denormalize_FC(x, amp_min, amp_max)

    dft_real = torch.ones(x.shape[0], img_shape * (img_shape // 2 + 1), dtype=real_part.dtype, device=real_part.device)
    dft_imag = torch.zeros(x.shape[0], img_shape * (img_shape // 2 + 1), dtype=imag_part.dtype, device=imag_part.device)
    
    dft_real[:, :real_part.shape[1]] = real_part
    dft_imag[:, :imag_part.shape[1]] = imag_part

    dft_real[:, dst_flatten_order] = torch.flatten(dft_real.clone(), start_dim=1)
    dft_imag[:, dst_flatten_order] = torch.flatten(dft_imag.clone(), start_dim=1)
    
    return dft_real.reshape(-1, img_shape, img_shape // 2 + 1), dft_imag.reshape(-1, img_shape, img_shape // 2 + 1)



def normalize_phi(phi):
    """
    Normalize phi to [-1, 1].

    :param phi:
    :return:
    """
    return phi / np.pi


def denormalize_phi(phi):
    """
    Invert `normalize_phi`.

    :param phi:
    :return:
    """
    return phi * np.pi


def normalize_amp(amp, amp_min, amp_max):
    """
    Normalize amplitudes to [-1, 1].

    :param amp:
    :param amp_min:
    :param amp_max:
    :return:
    """
    log_amps = log_amplitudes(amp)
    return 2 * (log_amps - amp_min) / (amp_max - amp_min) - 1


def denormalize_amp(amp, amp_min, amp_max):
    """
    Invert `normalize_amp`.

    :param amp:
    :param amp_min:
    :param amp_max:
    :return:
    """
    amp = (amp + 1) / 2.
    amp = (amp * (amp_max - amp_min)) + amp_min
    amp = torch.exp(amp)
    return amp


def fft_interpolate(srcx, srcy, dstx, dsty, src_fourier_coefficients, target_shape, dst_flatten_order):
    """
    Interpolates Fourier coefficients at (dstx, dsty) from Fourier coefficients in `sinogram_FC`.
    
    :param srcx: source x-coordinates
    :param srcy: source y-coordinates
    :param dstx: interpolated x-coordinates
    :param dsty: interolated y-coordinates
    :param src_fourier_coefficients: source Fourier coefficients
    :param target_shape: output shape of the interpolated Fourier coefficients
    :param dst_flatten_order: flattening order of (dstx, dsty)
    :return: 
    """
    vals = scipy.interpolate.griddata(
        (srcx, srcy),
        src_fourier_coefficients,
        (dstx, dsty),
        method='nearest',
        fill_value=1.0
    )
    output = np.zeros_like(vals)
    output[dst_flatten_order] = vals
    return output.reshape(target_shape)


def gaussian(x, mu, sig):
    """
    Compute Gaussian.

    :param x:
    :param mu:
    :param sig:
    :return:
    """
    return torch.exp(-torch.pow(x - mu, torch.tensor(2.)) / (2 * torch.pow(sig, torch.tensor(2.))))


def gaussian_psf(x, c, r):
    """
    Create 1D Gaussian shaped point spread function (PSF) profile normalized to [0, 1].

    :param x: range
    :param c: psf center
    :param r: psf sigma (radius)
    :return:
    """
    return torch.maximum((1 + gaussian(r, 0, r / 2.)) * gaussian(x, c, r / 2.) - gaussian(r, 0, r / 2.),
                         torch.zeros_like(x))


def psf_real(r, pixel_res=32):
    """
    Compute point spread function (PSF) with radius `r`.

    :param r: radius
    :param pixel_res: resolution in pixels (size of the PSF image)
    :return: PSF image
    """
    c = int(pixel_res / 2.)
    x, y = torch.meshgrid(torch.arange(pixel_res), torch.arange(pixel_res))
    psfimg = gaussian_psf(x, c, r) * gaussian_psf(y, c, r)
    psfimg = torch.roll(psfimg, -c, dims=0)
    psfimg = torch.roll(psfimg, -c, dims=1)
    psfimg /= torch.sum(psfimg)
    return psfimg


def psf_rfft(r, pixel_res=32):
    """
    Real FFT of PSF.

    :param r: radius of PSF
    :param pixel_res: resolution in pixel (size of PSF image)
    :return: rFFT of PSF
    """
    real=torch.real(torch.fft.rfftn(psf_real(r, pixel_res)))
    imag=torch.imag(torch.fft.rfftn(psf_real(r, pixel_res)))
    real=real.float()
    imag=imag.float()
    return real,imag


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
    beta = target.mean() - alpha * x.mean()
    return alpha * x + beta


def PSNR(gt, img, drange):
    """
    Computes the highest PSNR by affine rescaling of x,
    such that the mean squared error to gt is minimal.

    :param gt: ground truth
    :param img: image
    :param drange: data range
    :return: PSNR
    """
    img = normalize_minmse(img, gt)
    mse = torch.mean(torch.square(gt - img))
    return 20 * torch.log10(drange) - 10 * torch.log10(mse)


def normalize(data, mean, std):
    """
    Zero-mean, one standard dev. normalization

    :param data:
    :param mean:
    :param std:
    :return: normalized data
    """
    return (data - mean) / std


def denormalize(data, mean, std):
    """
    Invert `normalize`

    :param data:
    :param mean:
    :param std:
    :return: denormalized data
    """
    return (data * std) + mean


def pol2cart(r, phi):
    """
    Polar coordinates to cartesian coordinates.
    
    :param r: 
    :param phi: 
    :return: x, y
    """
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return (x, y)


def cart2pol(x, y):
    """
    Cartesian coordinates to polar coordinates.

    :param x:
    :param y:
    :return: r, phi
    """
    r = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    return r, phi

def apply_window_level(image, window_width, window_level):
    lower_bound = window_level - window_width / 2
    upper_bound = window_level + window_width / 2
    windowed_image = torch.clamp(image, min=lower_bound, max=upper_bound)
    windowed_image = (windowed_image - lower_bound) / (upper_bound - lower_bound)
    return windowed_image