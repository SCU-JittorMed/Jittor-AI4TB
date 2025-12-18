import os
os.environ["nvcc_flags"] = " -std=c++17 "
import jittor as jt
from jittor import nn, init
from jittor.dataset import Dataset
import numpy as np
import time
import csv

jt.flags.use_cuda = 1
jt.flags.use_cudnn = 1 

DATA_DIR = " s" 

BATCH_SIZE = 2
PATCH_SIZE = (96, 96, 96) 
EPOCHS = 20
LEARNING_RATE = 1e-3

class NNUnet3DDataset(Dataset):
    def __init__(self, data_dir, patch_size, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy') and '_seg' not in f]
        self.total_len = len(self.file_list)
        # 每个 Epoch 迭代次数多一点
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len * 20, shuffle=True)

    def __getitem__(self, index):
        case_id = self.file_list[index % len(self.file_list)]
        img_path = os.path.join(self.data_dir, case_id)
        seg_path = os.path.join(self.data_dir, case_id.replace('.npy', '_seg.npy'))
        
        # 加载 3D 数据
        image = np.load(img_path, mmap_mode='r') # (C, D, H, W) or (D, H, W)
        label = np.load(seg_path, mmap_mode='r')
        
        # 维度对齐
        if image.ndim == 3: image = image[None, ...]
        if label.ndim == 3: label = label[None, ...]
        
        C, D, H, W = image.shape
        crop_start = np.zeros(3, dtype=int)
        
        for attempt in range(20):
            # 生成随机起点
            for i in range(3):
                diff = image.shape[i+1] - self.patch_size[i]
                if diff > 0:
                    crop_start[i] = np.random.randint(0, diff + 1)
                else:
                    crop_start[i] = 0
            
            if attempt == 19 or np.random.rand() < 0.33:
                break
                
            # 检查这一块有没有肿瘤
            s, p = crop_start, self.patch_size
            temp_seg = label[:, s[0]:s[0]+p[0], s[1]:s[1]+p[1], s[2]:s[2]+p[2]]
            if np.sum(temp_seg) > 0:
                break # 找到了！

        s, p = crop_start, self.patch_size
        img_patch = image[:, s[0]:s[0]+p[0], s[1]:s[1]+p[1], s[2]:s[2]+p[2]]
        lbl_patch = label[:, s[0]:s[0]+p[0], s[1]:s[1]+p[1], s[2]:s[2]+p[2]]
        
        # Padding (如果原图比 Patch 小)
        if img_patch.shape[1] < p[0] or img_patch.shape[2] < p[1] or img_patch.shape[3] < p[2]:
            pad_d = max(0, p[0] - img_patch.shape[1])
            pad_h = max(0, p[1] - img_patch.shape[2])
            pad_w = max(0, p[2] - img_patch.shape[3])
            img_patch = np.pad(img_patch, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)), 'constant')
            lbl_patch = np.pad(lbl_patch, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)), 'constant')

        # 转 float32 并二值化标签
        return np.array(img_patch, dtype=np.float32), (lbl_patch > 0).astype(np.float32)

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 使用原生 Conv3d
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01)
        
    def execute(self, x):
        return self.act(self.norm(self.conv(x)))

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            ConvBlock3D(in_ch, out_ch, stride=stride),
            ConvBlock3D(out_ch, out_ch, stride=1)
        )
    def execute(self, x):
        return self.layer(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base=32):
        super().__init__()
        
        # Encoder (Downsampling)
        # stride=2 在 Conv 中实现下采样，比 MaxPool 更保留信息
        self.enc1 = DoubleConv3D(in_channels, base, stride=1)
        self.enc2 = DoubleConv3D(base, base*2, stride=2)
        self.enc3 = DoubleConv3D(base*2, base*4, stride=2)
        self.enc4 = DoubleConv3D(base*4, base*8, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(base*8, base*16, stride=2)
        
        # Decoder (Upsampling)
        self.up4 = nn.ConvTranspose3d(base*16, base*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(base*16, base*8) # concat后通道翻倍，需要缩回来
        
        self.up3 = nn.ConvTranspose3d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base*8, base*4)
        
        self.up2 = nn.ConvTranspose3d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base*4, base*2)
        
        self.up1 = nn.ConvTranspose3d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base*2, base)
        
        # Output
        self.final = nn.Conv3d(base, num_classes, kernel_size=1)

    def execute(self, x):
        # x: [B, C, D, H, W]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)
        
        u4 = self.up4(x5)
        # 简单对齐，防止尺寸不匹配
        if u4.shape != x4.shape: u4 = u4[:,:,:x4.shape[2],:x4.shape[3],:x4.shape[4]]
        d4 = self.dec4(jt.contrib.concat([u4, x4], dim=1))
        
        u3 = self.up3(d4)
        if u3.shape != x3.shape: u3 = u3[:,:,:x3.shape[2],:x3.shape[3],:x3.shape[4]]
        d3 = self.dec3(jt.contrib.concat([u3, x3], dim=1))
        
        u2 = self.up2(d3)
        if u2.shape != x2.shape: u2 = u2[:,:,:x2.shape[2],:x2.shape[3],:x2.shape[4]]
        d2 = self.dec2(jt.contrib.concat([u2, x2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape != x1.shape: u1 = u1[:,:,:x1.shape[2],:x1.shape[3],:x1.shape[4]]
        d1 = self.dec1(jt.contrib.concat([u1, x1], dim=1))
        
        return self.final(d1)

# ================= 4. 损失函数 (Dice + BCE) =================
# nnU-Net 标配：Combo Loss
def dc_ce_loss(outputs, targets):

    ce_loss = nn.binary_cross_entropy_with_logits(outputs, targets)
    
    # 2. Dice Loss (整体重叠度)
    probs = jt.sigmoid(outputs)
    
    # 平滑项
    smooth = 1e-5
    # 在 Batch 维度保留，其他维度求和
    intersection = (probs * targets).sum(dims=[2,3,4]) 
    union = probs.sum(dims=[2,3,4]) + targets.sum(dims=[2,3,4])
    
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    
    return ce_loss + dice_loss

if __name__ == "__main__":
    print(f"🔥 Jittor Native 3D UNet 训练启动...")
    print(f"📦 Patch Size: {PATCH_SIZE}, Batch Size: {BATCH_SIZE}")
    
    dataset = NNUnet3DDataset(DATA_DIR, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE)
    model = UNet3D(in_channels=1, num_classes=1, base=32) 
    optimizer = nn.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99, nesterov=True)
    
    csv_file = open("training_metrics_3d.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch", "Loss", "Time(s)", "Memory(MB)"])
    
    start_t = time.time()
    for epoch in range(EPOCHS):
        jt.sync_all()
        ep_start = time.time()
        ep_loss = 0
        steps = 0
        
        for i, (imgs, masks) in enumerate(dataset):
            # imgs: [B, 1, D, H, W]
            pred = model(imgs)
            loss = dc_ce_loss(pred, masks)
            
            optimizer.step(loss)
            ep_loss += loss.item()
            steps += 1
            
            if i % 5 == 0:
                print(f"Ep {epoch+1} Iter {i}: Loss={loss.item():.4f}")

        jt.sync_all()
        avg_loss = ep_loss / steps
        dur = time.time() - ep_start
        
        mem = jt.display_memory_info().total_used / 1024 / 1024
        
        print(f"✅ Ep {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {dur:.2f}s | Mem: {mem:.0f}MB")
        writer.writerow([epoch+1, avg_loss, dur, mem])
        csv_file.flush()
        
        model.save(f"unet3d_ep{epoch+1}.pkl")

    csv_file.close()
    print("🎉 3D 训练大功告成！")