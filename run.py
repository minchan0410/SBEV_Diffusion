import torch
import matplotlib.pyplot as plt
import os
import argparse
import math
from model import MNISTDiffusion

def load_robust_state_dict(model, ckpt_path, device):
    """
    접두어(module.)를 제거하고 가중치를 로드합니다.
    모델 구조가 달라도 일치하는 키만 로드하여 오류를 방지합니다(strict=False).
    """
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 1. model_ema가 있으면 우선 사용, 없으면 model 사용
    if 'model_ema' in checkpoint:
        state_dict = checkpoint['model_ema']
        print("Using EMA weights (Best quality).")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Using standard weights.")
    else:
        state_dict = checkpoint

    # 2. 키 이름 수정 (module. 제거)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 'module.' 접두사가 있으면 제거
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 3. 모델에 가중치 로드 (strict=False로 설정하여 일부 키가 안 맞아도 로드 진행)
    # 주의: Conditional 모델에 Unconditional 체크포인트를 넣으면 label_embedding이 초기화되지 않아 이상한 결과가 나올 수 있음
    model.load_state_dict(new_state_dict, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--digit', type=int, default=5, help='Digit to generate (0-9). Use -1 for all digits.')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples to generate per digit')
    parser.add_argument('--no_clip', action='store_true', help='Disable clipping')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화 (학습할 때 사용한 설정과 맞춰주세요)
    # 우리가 수정한 Conditional Model 구조입니다.
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,       # 학습 시 설정한 base_dim (기본: 64)
        dim_mults=[2, 4]   # 학습 시 설정한 dim_mults (기본: [2, 4])
    ).to(device)

    if not os.path.exists(args.ckpt):
        print(f"Error: 파일이 없습니다 -> {args.ckpt}")
        return

    # 가중치 로드
    model = load_robust_state_dict(model, args.ckpt, device)
    model.eval()

    # 생성할 숫자 라벨 설정
    if args.digit == -1:
        # 0부터 9까지 모두 생성
        target_digits = list(range(10))
        total_samples = 10
        print("Generating digits 0 to 9...")
    else:
        # 특정 숫자만 생성
        target_digits = [args.digit] * args.n_samples
        total_samples = len(target_digits)
        print(f"Generating {args.n_samples} images of number {args.digit}...")

    # 텐서 변환
    labels = torch.tensor(target_digits).long().to(device)

    # 추론 수행
    with torch.no_grad():
        # [중요] labels 인자를 전달해야 원하는 숫자가 나옵니다.
        # labels 개수(n_samples)만큼 이미지가 생성됩니다.
        samples = model.sampling(
            n_samples=total_samples,
            labels=labels,
            clipped_reverse_diffusion=not args.no_clip,
            device=device
        )

    # 결과 시각화 및 저장
    # samples shape: (N, 1, 28, 28) -> CPU로 이동 후 처리
    samples = samples.cpu().numpy()

    # 여러 장을 보여주기 위한 Plot 설정
    cols = 5 if total_samples > 1 else 1
    rows = math.ceil(total_samples / cols)
    
    plt.figure(figsize=(cols * 2, rows * 2))
    
    for i in range(total_samples):
        ax = plt.subplot(rows, cols, i + 1)
        img = samples[i].squeeze() # (1, 28, 28) -> (28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Digit: {target_digits[i]}")
        ax.axis('off')

    save_path = "generated_result.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"생성 완료! 결과가 {save_path}에 저장되었습니다.")
    # plt.show() # 필요시 주석 해제

if __name__ == "__main__":
    main()