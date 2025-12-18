import os
# 确保环境一致
os.environ["cc_path"] = "/usr/bin/g++-10"

import jittor as jt
from jittor import nn
import numpy as np
import nibabel as nib
import time
import math



jt.flags.use_cuda = 1
INPUT_PATH = ""
MODEL_PATH = " " 
OUTPUT_PATH = " "

BATCH_SIZE = 1 

# ================= 2D 模型定义 (保持不变) =================
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01)
    def execute(self, x): return self.act(self.norm(self.conv(x)))

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock2D(in_channels, base), ConvBlock2D(base, base))
        self.enc2 = nn.Sequential(ConvBlock2D(base, base*2, stride=2), ConvBlock2D(base*2, base*2))
        self.enc3 = nn.Sequential(ConvBlock2D(base*2, base*4, stride=2), ConvBlock2D(base*4, base*4))
        self.bottleneck = nn.Sequential(ConvBlock2D(base*4, base*8, stride=2), ConvBlock2D(base*8, base*8))
        
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock2D(base*8, base*4), ConvBlock2D(base*4, base*4))
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock2D(base*4, base*2), ConvBlock2D(base*2, base*2))
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock2D(base*2, base), ConvBlock2D(base, base))
        self.final = nn.Conv2d(base, num_classes, 1)

    def execute(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x_bot = self.bottleneck(x3)
        
        u3 = self.up3(x_bot)
        if u3.shape != x3.shape: u3 = u3[:, :, :x3.shape[2], :x3.shape[3]]
        d3 = self.dec3(jt.contrib.concat([u3, x3], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape != x2.shape: u2 = u2[:, :, :x2.shape[2], :x2.shape[3]]
        d2 = self.dec2(jt.contrib.concat([u2, x2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape != x1.shape: u1 = u1[:, :, :x1.shape[2], :x1.shape[3]]
        d1 = self.dec1(jt.contrib.concat([u1, x1], dim=1))
        return self.final(d1)

def preprocess_ct(img_data):
    img_data = np.clip(img_data, -1000, 1000)
    mean = np.mean(img_data)
    std = np.std(img_data)
    return (img_data - mean) / (std + 1e-8)

def predict_slices(model, volume, batch_size=1):
    D, H, W = volume.shape
    print(f"输入数据形状: {volume.shape} (切片数: {D})")
    
    predictions = np.zeros_like(volume)
    num_batches = math.ceil(D / batch_size)
    
    print(f"开始 2D 推理，Batch Size={batch_size}...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, D)
        
        # 1. 准备数据
        batch_slices = volume[start_idx:end_idx]
        batch_tensor = jt.array(batch_slices[:, None, :, :]).float()
        
        # 2. 推理
        with jt.no_grad():
            pred_batch = model(batch_tensor)
            pred_np = jt.sigmoid(pred_batch).numpy()
            
        # 3. 存结果
        predictions[start_idx:end_idx] = pred_np[:, 0, :, :]
        

        del batch_tensor, pred_batch, pred_np
        
        jt.gc()

        print(f"\r进度: {end_idx}/{D} slices", end='')
        
    print("\n推理完成！")
    return predictions

if __name__ == "__main__":

    jt.gc() 
    
    t_start_all = time.time()
    
    print(f"正在加载 2D 模型: {MODEL_PATH}")
    model = UNet2D() 
    
    if os.path.exists(MODEL_PATH):
        checkpoint = jt.load(MODEL_PATH)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 模型加载成功！")
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        exit()
        
    model.eval()

    print(f"正在读取 NIfTI: {INPUT_PATH}")
    nii_img = nib.load(INPUT_PATH)
    img_data = nii_img.get_fdata().astype(np.float32)
    affine = nii_img.affine
    
    if img_data.shape[0] == 512 and img_data.shape[1] == 512:
         img_data = img_data.transpose(2, 0, 1) 
    
    img_data = preprocess_ct(img_data)

    t_infer_start = time.time()
    prob_map = predict_slices(model, img_data, BATCH_SIZE)
    t_infer_end = time.time()
    
    print(f"⚡ 2D 纯推理耗时: {t_infer_end - t_infer_start:.2f}s")

    print("正在保存...")
    binary_mask = (prob_map > 0.5).astype(np.uint8)
    binary_mask = binary_mask.transpose(1, 2, 0)
    
    pred_nii = nib.Nifti1Image(binary_mask, affine)
    nib.save(pred_nii, OUTPUT_PATH)
    
    print(f"🎉 结果已保存: {OUTPUT_PATH}")