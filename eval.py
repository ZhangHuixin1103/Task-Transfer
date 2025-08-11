import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def eval_quality(gt_path, gen_path):
    gt_img = Image.open(gt_path).convert("RGB")
    gen_img = Image.open(gen_path).convert("RGB")
    gen_img = gen_img.resize(gt_img.size, Image.BICUBIC)
    gt_np = np.array(gt_img)
    pred_np = np.array(gen_img)

    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
    ssim = structural_similarity(gt_np, pred_np, channel_axis=-1, data_range=255)

    return psnr, ssim

psnr_1, ssim_1 = eval_quality("data/demo/deraining/2-derain.jpg", "data/demo/deraining/output-2-before.png")
psnr_2, ssim_2 = eval_quality("data/demo/deraining/2-derain.jpg", "data/demo/deraining/output-2-after.png")
print(f"PSNR: {psnr_1:.2f}, {psnr_2:.2f}; SSIM: {ssim_1:.4f}, {ssim_2:.4f}")
