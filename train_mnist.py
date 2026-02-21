import torch
import torch.nn as nn
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 디렉토리 내 모든 이미지 불러오기
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 201x101 이미지를 208x112로 만들기 위한 패딩
        # Pad 순서: (left, top, right, bottom) -> 가로 차이 11(5, 6), 세로 차이 7(3, 4)
        self.preprocess = transforms.Compose([
            transforms.Pad((5, 3, 6, 4), fill=255), # 배경이 흰색이면 fill=255 (검은색이면 0)
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # RGB 정규화
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # 컬러 이미지이므로 RGB로 변환
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        return image

def create_custom_dataloaders(batch_size, data_dir, num_workers=4):
    dataset = CustomImageDataset(root_dir=data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def parse_args():
    parser = argparse.ArgumentParser(description="Training Custom Diffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16) # 해상도가 커졌으므로 배치 사이즈 축소
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=16)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--log_freq', type=int, help='training log message printing frequence', default=10)
    parser.add_argument('--no_clip', action='store_true', help='set to normal sampling method without clip x_0')
    parser.add_argument('--cpu', action='store_true', help='cpu training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to your custom images directory')

    args = parser.parse_args()
    return args

def main(args):
    device = "cpu" if args.cpu else "cuda"
    train_dataloader = create_custom_dataloaders(batch_size=args.batch_size, data_dir=args.data_dir)
    
    # RGB(3채널) 및 패딩된 해상도(208, 112) 설정
    model = MNISTDiffusion(timesteps=args.timesteps,
                image_size=(208, 112),
                in_channels=3,
                base_dim=args.model_base_dim,
                dim_mults=[1, 2, 4, 8]).to(device)

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs*len(train_dataloader), pct_start=0.25, anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    for i in range(args.epochs):
        model.train()
        for j, image in enumerate(train_dataloader):
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            
            pred = model(image, noise)
            
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            if j % args.log_freq == 0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1, args.epochs, j, len(train_dataloader),
                                                                    loss.detach().cpu().item(), scheduler.get_last_lr()[0]))
        ckpt = {"model": model.state_dict(),
                "model_ema": model_ema.state_dict()}

        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, "results/steps_{:0>8}.pt".format(global_steps))

        model_ema.eval()
        
        # 라벨 인자 없이 샘플링 진행
        samples = model_ema.module.sampling(args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
        
        # 저장 전 208x112에서 원래 해상도인 201x101로 크롭 복원
        # 높이: 상단 3, 하단 4 제거 -> 3:-4
        # 너비: 좌측 5, 우측 6 제거 -> 5:-6
        samples = samples[:, :, 3:-4, 5:-6]
        
        save_image(samples, "results/steps_{:0>8}.png".format(global_steps), nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args = parse_args()
    main(args)