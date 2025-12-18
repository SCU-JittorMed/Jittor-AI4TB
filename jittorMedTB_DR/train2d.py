import os
# 确保使用 GCC 10
os.environ["cc_path"] = "/usr/bin/g++-10"

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset
import numpy as np
import time
import datetime
import json

# ================= 1. 2D 训练配置 =================
jt.flags.use_cuda = 1

DATA_DIR = " " 
SPLIT_JSON = " "
FOLD = 0 

# 🔥 2D 配置变化
BATCH_SIZE = 12  # 2D 比较小，可以开大 Batch
PATCH_SIZE = (512, 512) # 2D 训练通常是全分辨率切片或大 Patch
EPOCHS = 1000
LEARNING_RATE = 1e-3

# 保存文件名区分
LOG_FILE = "training_log_2d.txt"
CHECKPOINT_DIR = "./checkpoints_2d"

if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

# ================= 2. 日志函数 =================
def print_log(msg, f_handle=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    content = f"{timestamp}: {msg}"
    print(content)
    if f_handle:
        f_handle.write(content + "\n")
        f_handle.flush()

def print_newline_log(f_handle=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    content = f"{timestamp}: "
    print(content)
    if f_handle:
        f_handle.write(content + "\n")
        f_handle.flush()

# ================= 3. 2D 数据集 (核心修改) =================
class NNUnet2DDataset(Dataset):
    def __init__(self, data_dir, split_json, fold, is_train=True, patch_size=(512, 512), batch_size=12, log_f=None):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.is_train = is_train
        
        with open(split_json, 'r') as f:
            splits = json.load(f)
            
        if fold >= len(splits): raise ValueError(f"Fold {fold} 不存在")
        current_split = splits[fold]
        target_keys = current_split['train'] if is_train else current_split['val']
        
        msg = f"[{'Train' if is_train else 'Val'}] 加载 Fold {fold}, 共 {len(target_keys)} 个病例 (2D模式)"
        if log_f: print_log(msg, log_f)

        self.file_list = []
        all_files = os.listdir(data_dir)
        for fname in all_files:
            if fname.endswith('.npy') and '_seg' not in fname:
                case_id = fname.replace('.npy', '')
                if case_id in target_keys:
                    self.file_list.append(fname)
        
        # 训练集每个 epoch 迭代次数
        self.total_len = 250 * batch_size if is_train else len(self.file_list)
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len, shuffle=is_train)

    def __getitem__(self, index):
        idx = index % len(self.file_list)
        case_id = self.file_list[idx]
        img_path = os.path.join(self.data_dir, case_id)
        seg_path = os.path.join(self.data_dir, case_id.replace('.npy', '_seg.npy'))
        
        # 读取 3D 数据
        image_3d = np.load(img_path, mmap_mode='r') 
        label_3d = np.load(seg_path, mmap_mode='r') # Shape: [C, D, H, W]
        
        C, D, H, W = image_3d.shape
        
        # 🔥 核心策略：随机选择一个切片 Z
        selected_z = 0
        
        if self.is_train:
            # 训练时：33% 的概率强制选中有标签的前景层，避免一直在学背景
            if np.random.rand() < 0.33:
                # 寻找有标签的层
                foreground_slices = np.where(np.sum(label_3d[0], axis=(1,2)) > 0)[0]
                if len(foreground_slices) > 0:
                    selected_z = np.random.choice(foreground_slices)
                else:
                    selected_z = np.random.randint(0, D)
            else:
                selected_z = np.random.randint(0, D)
        else:
            # 验证时：简单取中间层或随机层 (验证通常应该在 3D 上做，这里简化为随机抽 2D 层验证 loss)
            selected_z = np.random.randint(0, D)

        # 提取 2D 切片 [C, H, W]
        image_2d = image_3d[:, selected_z, :, :]
        label_2d = label_3d[:, selected_z, :, :]
        
        # 2D 随机裁剪
        h_idx = np.random.randint(0, max(1, H - self.patch_size[0]))
        w_idx = np.random.randint(0, max(1, W - self.patch_size[1]))
        
        img_patch = image_2d[:, h_idx:h_idx+self.patch_size[0], w_idx:w_idx+self.patch_size[1]]
        lbl_patch = label_2d[:, h_idx:h_idx+self.patch_size[0], w_idx:w_idx+self.patch_size[1]]
        
        # Padding (如果切片比 patch 小)
        if img_patch.shape[1] < self.patch_size[0] or img_patch.shape[2] < self.patch_size[1]:
            pad = [(0,0)] + [(0, max(0, self.patch_size[i]-img_patch.shape[i+1])) for i in range(2)]
            img_patch = np.pad(img_patch, pad, 'constant')
            lbl_patch = np.pad(lbl_patch, pad, 'constant')

        return np.array(img_patch, dtype=np.float32), (lbl_patch > 0).astype(np.float32)

# ================= 4. 模型 (2D U-Net) =================
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 🔥 改为 Conv2d
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        # 🔥 改为 InstanceNorm2d
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01)
    def execute(self, x): return self.act(self.norm(self.conv(x)))

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base=32):
        super().__init__()
        # 结构逻辑和 3D 一样，只是算子换成 2D
        self.enc1 = nn.Sequential(ConvBlock2D(in_channels, base), ConvBlock2D(base, base))
        self.enc2 = nn.Sequential(ConvBlock2D(base, base*2, 2), ConvBlock2D(base*2, base*2))
        self.enc3 = nn.Sequential(ConvBlock2D(base*2, base*4, 2), ConvBlock2D(base*4, base*4))
        self.bottleneck = nn.Sequential(ConvBlock2D(base*4, base*8, 2), ConvBlock2D(base*8, base*8))
        
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
        if u3.shape!=x3.shape: u3=u3[:,:,:x3.shape[2],:x3.shape[3]] # 2D 只有 H, W
        d3 = self.dec3(jt.contrib.concat([u3, x3], dim=1))
        u2 = self.up2(d3)
        if u2.shape!=x2.shape: u2=u2[:,:,:x2.shape[2],:x2.shape[3]]
        d2 = self.dec2(jt.contrib.concat([u2, x2], dim=1))
        u1 = self.up1(d2)
        if u1.shape!=x1.shape: u1=u1[:,:,:x1.shape[2],:x1.shape[3]]
        d1 = self.dec1(jt.contrib.concat([u1, x1], dim=1))
        return self.final(d1)

# ================= 5. Loss (2D版) =================
def soft_dice_loss(outputs, targets):
    probs = jt.sigmoid(outputs)
    # 🔥 2D 只在 [2, 3] 维度求和 (Batch, Channel, H, W)
    inter = (probs * targets).sum(dims=[2,3])
    union = probs.sum(dims=[2,3]) + targets.sum(dims=[2,3])
    dice = (2. * inter + 1e-5) / (union + 1e-5)
    return -dice.mean()

def calculate_dice(outputs, targets):
    probs = (jt.sigmoid(outputs) > 0.5).float()
    inter = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    return (2. * inter + 1e-5) / (union + 1e-5)

# ================= 6. 主程序 =================
if __name__ == "__main__":
    log_file = open(LOG_FILE, "a")
    
    print_log(f"Loading 2D Training Task...", log_file)
    train_ds = NNUnet2DDataset(DATA_DIR, SPLIT_JSON, FOLD, is_train=True, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, log_f=log_file)
    val_ds = NNUnet2DDataset(DATA_DIR, SPLIT_JSON, FOLD, is_train=False, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, log_f=log_file)
    
    model = UNet2D()
    optimizer = nn.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99, nesterov=True)
    
    # 断点续训逻辑 (针对 2D checkpoint)
    start_epoch = 0
    best_dice = 0.0
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_2d_latest.pkl")
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_2d_best.pkl")
    
    if os.path.exists(latest_ckpt_path):
        print_log(f"⚠️ Found checkpoint: {latest_ckpt_path}, Resuming...", log_file)
        checkpoint = jt.load(latest_ckpt_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        print_log(f"✅ Resumed from Epoch {start_epoch-1}. Best Dice: {best_dice:.4f}", log_file)
    else:
        print_log("🚀 Starting fresh 2D training.", log_file)

    for epoch in range(start_epoch, EPOCHS):
        print_newline_log(log_file)
        print_log(f"Epoch {epoch}", log_file)
        
        optimizer.lr = LEARNING_RATE * (1 - epoch/EPOCHS)**0.9
        print_log(f"Current learning rate: {optimizer.lr:.4f}", log_file)
        
        ep_start = time.time()
        train_loss_list = []
        model.train()
        for imgs, masks in train_ds:
            pred = model(imgs)
            loss = soft_dice_loss(pred, masks)
            optimizer.step(loss)
            train_loss_list.append(loss.item())
        
        val_loss_list = []
        dice_list = []
        model.eval()
        with jt.no_grad():
            for imgs, masks in val_ds:
                pred = model(imgs)
                v_loss = soft_dice_loss(pred, masks)
                dice = calculate_dice(pred, masks)
                val_loss_list.append(v_loss.item())
                dice_list.append(dice.item())
        
        avg_train_loss = np.mean(train_loss_list)
        avg_val_loss = np.mean(val_loss_list)
        avg_dice = np.mean(dice_list)
        dur = time.time() - ep_start
        
        print_log(f"train_loss {avg_train_loss:.4f}", log_file)
        print_log(f"val_loss {avg_val_loss:.4f}", log_file)
        print_log(f"Pseudo dice [{avg_dice:.4f}]", log_file)
        print_log(f"Epoch time: {dur:.2f} s", log_file)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_dice': best_dice
        }
        
        jt.save(checkpoint_data, latest_ckpt_path)
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            print_log(f"⭐ New Best Dice: {best_dice:.4f}", log_file)
            jt.save(checkpoint_data, best_ckpt_path)
        
        if (epoch + 1) % 50 == 0:
            archive_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_2d_ep{epoch}.pkl")
            jt.save(checkpoint_data, archive_path)
            
    log_file.close()