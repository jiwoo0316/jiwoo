import numpy as np
import torch
from torch.utils.data import Dataset


class LoRaResearchDataset(Dataset):
    """
    feature_type에 따라
      - complex 입력 (2채널)
      - mag 입력 (1채널)
    을 생성할 수 있도록 만든 데이터셋.

    [개선 1] 매 샘플마다 채널 파라미터를 랜덤화하여
    다양한 CFO/Multipath 환경을 학습에 반영함.
    """

    def __init__(
        self,
        simulator,
        num_samples: int,
        snr_range,
        impairment_config: dict,
        mode: str = "train",
        feature_type: str = "complex",
        randomize_channel: bool = False,
    ):
        self.simulator = simulator
        self.num_samples = num_samples
        self.feature_type = feature_type
        self.randomize_channel = randomize_channel

        print(f"[{mode.upper()} | {feature_type.upper()}] {num_samples}개의 데이터 생성 중...")
        self.data_x = []
        self.data_y = []

        for i in range(num_samples):
            # eval 모드에서는 내부 seed를 고정해 평가 표본을 재현 가능하게 함
            if mode == "eval":
                np.random.seed(2026 + i)

            # 랜덤 label 선택
            label = np.random.randint(0, self.simulator.M)

            # clean signal 생성
            clean_sig = self.simulator.generate_symbol(label)

            # snr_range가 tuple이면 구간 내 랜덤 SNR, 아니면 고정 SNR
            target_snr = (
                np.random.uniform(snr_range[0], snr_range[1])
                if isinstance(snr_range, tuple)
                else snr_range
            )

            # [개선 1] 훈련 시 매 샘플마다 채널 파라미터 랜덤화
            if self.randomize_channel and mode == "train":
                sample_config = self._random_impairment_config()
            else:
                sample_config = impairment_config

            # 채널 왜곡 적용
            noisy_sig = self.simulator.apply_impaired_channel(clean_sig, target_snr, sample_config)

            # feature_type에 따라 특징 추출 방식 선택
            if self.feature_type == "complex":
                features = self.simulator.dechirp_and_fft_complex(noisy_sig)
            else:
                features = self.simulator.dechirp_and_fft_mag(noisy_sig)

            self.data_x.append(torch.tensor(features, dtype=torch.float32))
            self.data_y.append(torch.tensor(label, dtype=torch.long))

    @staticmethod
    def _random_impairment_config():
        """
        매 샘플마다 호출되어 완전히 랜덤한 채널 환경을 생성함.

        - CFO: ±0.1 ~ ±0.5 bin 범위에서 균일 랜덤
        - Multipath: tap 수 2~4개, 크기/위상 랜덤, delay 랜덤
        - 일부 확률로 AWGN-only 환경도 섞어줌 (20%)
        """
        # 20% 확률로 깨끗한 AWGN-only
        if np.random.rand() < 0.2:
            return {"use_cfo": False, "use_multipath": False}

        # CFO 랜덤화
        use_cfo = np.random.rand() < 0.8  # 80% 확률로 CFO 사용
        max_cfo_bins = np.random.uniform(0.05, 0.5) if use_cfo else 0.0

        # Multipath 랜덤화
        use_multipath = np.random.rand() < 0.8  # 80% 확률로 multipath 사용
        taps = [1.0]
        delays = [0]

        if use_multipath:
            num_extra_taps = np.random.randint(1, 4)  # 반사파 1~3개
            for _ in range(num_extra_taps):
                mag = np.random.uniform(0.1, 0.7)
                phase = np.random.uniform(0, 2 * np.pi)
                taps.append(mag * np.exp(1j * phase))

            # delay는 1~12 범위에서 겹치지 않게 선택
            extra_delays = sorted(
                np.random.choice(range(1, 13), num_extra_taps, replace=False).tolist()
            )
            delays.extend(extra_delays)

        return {
            "use_cfo": use_cfo,
            "max_cfo_bins": max_cfo_bins,
            "use_multipath": use_multipath,
            "multipath_taps": taps,
            "multipath_delays": delays,
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]