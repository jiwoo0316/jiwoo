import os
import torch
from torch.utils.data import DataLoader, random_split

from utils import set_seed
from simulator import LoRaResearchSimulator
from dataset import LoRaResearchDataset
from models import LoRaCNN
from training import train_research_model
from evaluation import evaluate_ablation_model


def main():
    set_seed()
    os.makedirs("saved_models", exist_ok=True)

    # LoRa-like simulator 생성
    sim = LoRaResearchSimulator(sf=7, bw=125e3, fs=1e6)

    # 두 모델 초기화
    model_comp = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=2)
    model_mag = LoRaCNN(num_classes=sim.M, input_length=sim.N, in_channels=1)

    # [개선 1] 파일명을 바꿔서 기존 모델과 구분
    path_comp = "saved_models/lora_comp_cnn_v3.pth"
    path_mag = "saved_models/lora_mag_cnn_v3.pth"

    # [개선 1] 훈련 시 기본 config (randomize_channel=True일 때는 참고용)
    train_config = {"use_cfo": True, "max_cfo_bins": 0.35, "use_multipath": True}

    # [개선 1] SNR 범위를 (-25, 0)으로 확대 (기존: -20, 0)
    train_snr_range = (-25, 0)
    total_samples = 40000

    # 1. Complex CNN 훈련 또는 로드
    if os.path.exists(path_comp):
        model_comp.load_state_dict(torch.load(path_comp, map_location=torch.device("cpu")))
    else:
        print(">> [학습 1/2] Complex CNN 훈련 중...")
        ds_comp = LoRaResearchDataset(
            sim,
            total_samples,
            train_snr_range,
            train_config,
            feature_type="complex",
            randomize_channel=True,  # [개선 1] 채널 랜덤화 활성화
        )
        dl_train, dl_val = random_split(
            ds_comp,
            [int(0.85 * total_samples), total_samples - int(0.85 * total_samples)],
        )
        model_comp = train_research_model(
            model_comp,
            DataLoader(dl_train, batch_size=256, shuffle=True),
            DataLoader(dl_val, batch_size=256),
            num_epochs=25,
        )
        torch.save(model_comp.state_dict(), path_comp)

    # 2. Mag CNN 훈련 또는 로드
    if os.path.exists(path_mag):
        model_mag.load_state_dict(torch.load(path_mag, map_location=torch.device("cpu")))
    else:
        print("\n>> [학습 2/2] Mag CNN (Ablation) 훈련 중...")
        ds_mag = LoRaResearchDataset(
            sim,
            total_samples,
            train_snr_range,
            train_config,
            feature_type="mag",
            randomize_channel=True,  # [개선 1] 채널 랜덤화 활성화
        )
        dl_train, dl_val = random_split(
            ds_mag,
            [int(0.85 * total_samples), total_samples - int(0.85 * total_samples)],
        )
        model_mag = train_research_model(
            model_mag,
            DataLoader(dl_train, batch_size=256, shuffle=True),
            DataLoader(dl_val, batch_size=256),
            num_epochs=25,
        )
        torch.save(model_mag.state_dict(), path_mag)

    # 평가할 SNR 점들
    test_snrs = list(range(-25, 1, 2))

    # Scenario A: Pure AWGN
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        {"use_cfo": False, "use_multipath": False},
        "Scenario A - Pure AWGN",
    )

    # Scenario B: Seen Impaired
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        train_config,
        "Scenario B - Seen Impaired",
    )

    # Scenario C: Unseen Impaired
    config_unseen = {
        "use_cfo": True,
        "max_cfo_bins": 0.45,
        "use_multipath": True,
        "multipath_taps": [0.9, -0.6j, 0.4],
        "multipath_delays": [0, 4, 9],
    }
    evaluate_ablation_model(
        model_comp,
        model_mag,
        sim,
        test_snrs,
        config_unseen,
        "Scenario C - Unseen Impaired",
    )

    print("\n========== [완료] ==========")


if __name__ == "__main__":
    main()