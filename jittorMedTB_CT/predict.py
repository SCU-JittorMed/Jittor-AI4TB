import os
# 确保环境一致
os.environ["cc_path"] = "/usr/bin/g++-10"

import jittor as jt
from jittor import nn
import numpy as np
import nibabel as nib
import time
import math

# ================= 配置 =================
jt.flags.use_cuda = 1

# 输入文件
INPUT_PATH = " "
 
MODEL_PATH = " " 

OUTPUT_PATH = " "

# 和训练保持一致
PATCH_SIZE = (96, 96, 96) 
STRIDE = (48, 48, 48) 

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01)
    def execute(self, x): return self.act(self.norm(self.conv(x)))

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock3D(in_channels, base), ConvBlock3D(base, base))
        self.enc2 = nn.Sequential(ConvBlock3D(base, base*2, 2), ConvBlock3D(base*2, base*2))
        self.enc3 = nn.Sequential(ConvBlock3D(base*2, base*4, 2), ConvBlock3D(base*4, base*4))
        self.bottleneck = nn.Sequential(ConvBlock3D(base*4, base*8, 2), ConvBlock3D(base*8, base*8))
        
        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock3D(base*8, base*4), ConvBlock3D(base*4, base*4))
        
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock3D(base*4, base*2), ConvBlock3D(base*2, base*2))
        
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock3D(base*2, base), ConvBlock3D(base, base))
        
        self.final = nn.Conv3d(base, num_classes, 1)

    def execute(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x_bot = self.bottleneck(x3)
        
        u3 = self.up3(x_bot)
        if u3.shape!=x3.shape: u3=u3[:,:,:x3.shape[2],:x3.shape[3],:x3.shape[4]]
        d3 = self.dec3(jt.contrib.concat([u3, x3], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape!=x2.shape: u2=u2[:,:,:x2.shape[2],:x2.shape[3],:x2.shape[4]]
        d2 = self.dec2(jt.contrib.concat([u2, x2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape!=x1.shape: u1=u1[:,:,:x1.shape[2],:x1.shape[3],:x1.shape[4]]
        d1 = self.dec1(jt.contrib.concat([u1, x1], dim=1))
        
        return self.final(d1)

# ================= 2. 3D 滑窗推理函数 =================
def predict_sliding_window(model, image, patch_size, stride):
    D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # 结果容器
    prediction_map = np.zeros(image.shape, dtype=np.float32)
    count_map = np.zeros(image.shape, dtype=np.float32)
    
    # 计算步数
    dz = math.ceil((D - pd) / sd) + 1
    dy = math.ceil((H - ph) / sh) + 1
    dx = math.ceil((W - pw) / sw) + 1
    total = dz * dy * dx
    
    print(f"开始 3D 滑窗推理 (Patch={patch_size})... 总窗口数: {total}")
    
    cnt = 0
    # 强制 GC 避免显存碎片
    jt.gc()
    
    for z in range(0, D - pd + sd, sd):
        z = min(z, D - pd)
        for y in range(0, H - ph + sh, sh):
            y = min(y, H - ph)
            for x in range(0, W - pw + sw, sw):
                x = min(x, W - pw)
                
                # 1. 切块
                patch = image[z:z+pd, y:y+ph, x:x+pw]
                

                patch_tensor = jt.array(patch[None, None, ...])
                
                # 3. 推理
                with jt.no_grad():
                    pred = model(patch_tensor)
                    pred = jt.sigmoid(pred)
                    pred_np = pred.numpy()[0, 0] # 取出 [D, H, W]
                
                # 4. 累加
                prediction_map[z:z+pd, y:y+ph, x:x+pw] += pred_np
                count_map[z:z+pd, y:y+ph, x:x+pw] += 1.0
                
                cnt += 1
                if cnt % 5 == 0:
                    print(f"\r进度: {cnt}/{total}", end='')
                    jt.gc()
    
    print("\n拼接完成，归一化中...")
    return prediction_map / count_map

# ================= 3. 主程序 =================
if __name__ == "__main__":
    t_start = time.time()
    
    print(f"正在加载模型: {MODEL_PATH}")
    model = UNet3D()
    
    if os.path.exists(MODEL_PATH):
        checkpoint = jt.load(MODEL_PATH)
        try:
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                print("✅ 成功加载 model_state！")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 成功加载参数字典！")
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            print("请检查你的 MODEL_PATH 是否指向了正确的 3D 训练存档！")
            exit()
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        print("请修改脚本中的 MODEL_PATH 变量！")
        exit()
        
    model.eval()
    
    print(f"读取数据: {INPUT_PATH}")
    nii = nib.load(INPUT_PATH)
    data = nii.get_fdata().astype(np.float32)
    

    data = np.clip(data, -1000, 1000)
    data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    print("🚀 开始预测...")
    t_infer_start = time.time()
    
    prob_map = predict_sliding_window(model, data, PATCH_SIZE, STRIDE)
    
    t_infer_end = time.time()
    print(f"⚡ 推理耗时: {t_infer_end - t_infer_start:.2f}s")
    
    print("保存结果...")
    seg_map = (prob_map > 0.5).astype(np.uint8)
    nib.save(nib.Nifti1Image(seg_map, nii.affine), OUTPUT_PATH)
    print(f"🎉 完成！结果在: {OUTPUT_PATH}")