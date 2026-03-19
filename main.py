import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import time
import os
from scipy.signal import lfilter

# ============================================================
# 0. 실험 재현성 확보
# ============================================================
def set_seed(seed=2026):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ============================================================
# 1. Baseline 시뮬레이터
# ============================================================
class LoRaResearchSimulator:
    def __init__(self, sf=7, bw=125e3, fs=1e6):
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.M = 2**sf                
        self.Ts = self.M / bw         
        self.N = int(self.Ts * fs)    
        self.osr = self.N // self.M 
        
        # 이산시간(Discrete-time) 위상 연속 첩(Chirp) 신호 생성
        t = np.arange(self.N) / self.fs
        self.base_phase = np.pi * (self.bw**2 / self.M) * (t**2)
        self.downchirp = np.exp(-1j * self.base_phase)

    def generate_symbol(self, symbol):
        # 정답 빈(Bin)이 정확히 symbol * osr 위치에 오도록 톤(Tone) 합성
        n = np.arange(self.N)
        tone = np.exp(1j * 2 * np.pi * (symbol * self.osr) * n / self.N)
        return np.exp(1j * self.base_phase) * tone

    def apply_impaired_channel(self, signal, snr_db, impairment_config):
        impaired_signal = np.copy(signal)
        
        if impairment_config.get('use_multipath', False):
            taps = impairment_config.get('multipath_taps', [1.0, 0.4j, 0.2])
            delays = impairment_config.get('multipath_delays', [0, 2, 5]) 
            h = np.zeros(max(delays) + 1, dtype=np.complex128)
            h[delays] = taps
            impaired_signal = lfilter(h, 1, impaired_signal)
            
        if impairment_config.get('use_cfo', False):
            bin_spacing = self.bw / self.M 
            max_cfo_bins = impairment_config.get('max_cfo_bins', 0.35)
            cfo_hz = np.random.uniform(-max_cfo_bins, max_cfo_bins) * bin_spacing 
            cfo_phase = 2 * np.pi * cfo_hz * (np.arange(self.N) / self.fs)
            impaired_signal = impaired_signal * np.exp(1j * cfo_phase)

        signal_power = np.mean(np.abs(impaired_signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(self.N) + 1j * np.random.randn(self.N))
        
        return impaired_signal + noise

    def dechirp_and_fft_complex(self, iq_signal):
        dechirped = iq_signal * self.downchirp
        fft_complex = np.fft.fft(dechirped)
        max_val = np.max(np.abs(fft_complex)) + 1e-10
        fft_norm = fft_complex / max_val
        return np.vstack((np.real(fft_norm), np.imag(fft_norm))) # (2, N) 채널

    # --------------------------------------------------------
    # OSR 보정 베이스라인 복조기
    # --------------------------------------------------------
    def baseline_demod_naive(self, rx_signal):
        dechirped = rx_signal * self.downchirp
        fft_mag = np.abs(np.fft.fft(dechirped))
        peak_idx = int(np.argmax(fft_mag))
        return int(np.round(peak_idx / self.osr)) % self.M

    def baseline_demod_grouped_bin(self, rx_signal, window_size=2):
        dechirped = rx_signal * self.downchirp
        fft_mag_sq = np.abs(np.fft.fft(dechirped))**2
        
        grouped_energy = np.zeros(self.M)
        for k in range(self.M):
            center = int(np.round(k * self.osr))
            indices = np.mod(np.arange(center - window_size, center + window_size + 1), self.N)
            grouped_energy[k] = np.sum(fft_mag_sq[indices])
            
        return int(np.argmax(grouped_energy))

# ============================================================
# 2. 검증 기능이 포함된 Dataset
# ============================================================
class LoRaResearchDataset(Dataset):
    def __init__(self, simulator, num_samples, snr_range, impairment_config, mode='train'):
        self.simulator = simulator
        self.num_samples = num_samples
        
        print(f"[{mode.upper()}] {num_samples}개의 데이터를 생성 중.")
        self.data_x = []
        self.data_y = []
        
        for i in range(num_samples):
            if mode == 'eval': np.random.seed(2026 + i)
            label = np.random.randint(0, self.simulator.M)
            clean_sig = self.simulator.generate_symbol(label)
            target_snr = np.random.uniform(snr_range[0], snr_range[1]) if isinstance(snr_range, tuple) else snr_range
            
            noisy_sig = self.simulator.apply_impaired_channel(clean_sig, target_snr, impairment_config)
            fft_features = self.simulator.dechirp_and_fft_complex(noisy_sig)
            
            self.data_x.append(torch.tensor(fft_features, dtype=torch.float32))
            self.data_y.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

# ============================================================
# 3. 2-Channel Complex Input 1D CNN
# ============================================================
class LoRaComplexInputCNN(nn.Module):
    def __init__(self, num_classes, input_length):
        super(LoRaComplexInputCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, 2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2, 2)
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, 2, input_length)
            dummy_x = self.pool1(F.relu(self.bn1(self.conv1(dummy_x))))
            dummy_x = self.pool2(F.relu(self.bn2(self.conv2(dummy_x))))
            dummy_x = self.pool3(F.relu(self.bn3(self.conv3(dummy_x))))
            dummy_x = F.relu(self.bn4(self.conv4(dummy_x)))
            flattened_size = dummy_x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================
# 4. 검증 및 Best Checkpoint 학습 루프
# ============================================================
def train_research_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print(f"\n[학습 시작] Device: {device}, Epochs: {num_epochs}")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / total) * 100
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch [{epoch+1}/{num_epochs}] -> Best Model Saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
    print(f"\n가장 낮은 Val Loss({best_val_loss:.4f}) 모델을 복원합니다.")
    model.load_state_dict(best_model_state)
    return model

# ============================================================
# 5. 대규모 몬테카를로 정밀 평가 및 3종 베이스라인 비교
# ============================================================
def evaluate_model(model, simulator, snr_list, impairment_config, benchmark_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    
    cnn_ser, naive_ser, grouped_ser = [], [], []
    
    print(f"\n[정밀 평가 시작 : {benchmark_name}]")
    print("-" * 80)
    
    for snr in snr_list:
        # 저-SNR에서는 샘플을 늘려 통계적 신뢰도 확보
        if snr >= -10: num_test_samples = 50000 
        elif snr >= -15: num_test_samples = 10000
        else: num_test_samples = 3000
            
        print(f"SNR {snr:3d} dB 평가 중. (Monte Carlo 샘플: {num_test_samples})")
        
        c_correct, n_correct, g_correct = 0, 0, 0
        batch_size = 1000
        num_batches = num_test_samples // batch_size
        
        with torch.no_grad():
            for _ in range(num_batches):
                labels_np = np.random.randint(0, simulator.M, batch_size)
                rx_signals_np = []
                features_cnn = []
                
                for lbl in labels_np:
                    clean_sig = simulator.generate_symbol(lbl)
                    noisy_sig = simulator.apply_impaired_channel(clean_sig, snr, impairment_config)
                    rx_signals_np.append(noisy_sig)
                    features_cnn.append(torch.tensor(simulator.dechirp_and_fft_complex(noisy_sig), dtype=torch.float32))
                
                # Baseline 채점
                for i in range(batch_size):
                    if simulator.baseline_demod_naive(rx_signals_np[i]) == labels_np[i]:
                        n_correct += 1
                    if simulator.baseline_demod_grouped_bin(rx_signals_np[i], window_size=2) == labels_np[i]:
                        g_correct += 1
                
                # CNN 채점
                feature_tensor = torch.stack(features_cnn).to(device)
                labels_tensor = torch.tensor(labels_np, dtype=torch.long).to(device)
                output = model(feature_tensor)
                _, predicted = torch.max(output.data, 1)
                c_correct += (predicted == labels_tensor).sum().item()
            
        cnn_ser.append(1.0 - (c_correct / num_test_samples))
        naive_ser.append(1.0 - (n_correct / num_test_samples))
        grouped_ser.append(1.0 - (g_correct / num_test_samples))
        
        print(f"  -> SER [CNN]: {cnn_ser[-1]:.5f} | [Grouped]: {grouped_ser[-1]:.5f} | [Naive]: {naive_ser[-1]:.5f}")
            
    # 시각화
    filename = f"ser_curve_{benchmark_name.lower().replace(' ', '_')}.png"
    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_list, naive_ser, marker='x', linestyle=':', color='gray', alpha=0.7, label='Naive FFT (Corrected OSR)')
    plt.semilogy(snr_list, grouped_ser, marker='s', linestyle='--', color='black', label='Grouped-Bin Classical (Fair Baseline)')
    plt.semilogy(snr_list, cnn_ser, marker='o', linestyle='-', color='red', linewidth=2, label='Proposed 2-Ch CNN')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12)
    plt.ylabel('Symbol Error Rate (SER)', fontsize=12)
    plt.title(f'LoRa Demodulation Performance: {benchmark_name}', fontsize=14)
    plt.ylim([5e-5, 1.1])
    plt.legend(fontsize=11, loc='lower left')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return cnn_ser, grouped_ser

# ============================================================
# 6. 메인 실행 블록
# ============================================================
if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    sim = LoRaResearchSimulator(sf=7, bw=125e3, fs=1e6)
    
    # 0. 노이즈 없는 환경에서 Baseline 테스트
    print("\n노이즈 없는 환경에서 Baseline 로직 검증 중.")
    sanity_n, sanity_g = 0, 0
    for i in range(sim.M):
        clean_sig = sim.generate_symbol(i)
        if sim.baseline_demod_naive(clean_sig) == i: sanity_n += 1
        if sim.baseline_demod_grouped_bin(clean_sig, window_size=2) == i: sanity_g += 1
    
    print(f"Naive Baseline: {sanity_n}/{sim.M} 정답 | Grouped Baseline: {sanity_g}/{sim.M} 정답")
    if sanity_n != sim.M or sanity_g != sim.M:
        raise ValueError("경고: Baseline 구현에 오류가 있습니다. [정답률 100% 미만]")
    print("Baseline 로직 정상 작동 확인 완료.\n")

    # 1. 모델 초기화 및 가중치 파일 확인
    dummy_x = torch.zeros(1, 2, sim.N)
    model = LoRaComplexInputCNN(num_classes=sim.M, input_length=sim.N)
    
    model_path = 'saved_models/lora_complex_cnn_best.pth'
    if os.path.exists(model_path):
        print(">> 저장된 모델 가중치를 발견했습니다. 학습을 건너뛰고 평가를 진행합니다.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        best_trained_model = model
    else:
        print(">> 저장된 모델이 없습니다. 새롭게 모델 학습을 시작합니다.")
        train_config = {'use_cfo': True, 'max_cfo_bins': 0.35, 'use_multipath': True}
        total_samples = 40000 
        dataset = LoRaResearchDataset(sim, total_samples, (-20, 0), train_config)
        train_size = int(0.85 * total_samples)
        train_dataset, val_dataset = random_split(dataset, [train_size, total_samples - train_size])
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        best_trained_model = train_research_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.0003)
        torch.save(best_trained_model.state_dict(), model_path)

    # 2. 다중 시나리오 벤치마크 평가 
    test_snrs = np.arange(-25, 1, 2)
    
    # 시나리오 A: Pure AWGN
    config_awgn = {'use_cfo': False, 'use_multipath': False}
    evaluate_model(best_trained_model, sim, test_snrs, config_awgn, "Scenario A - Pure AWGN")
    
    # 시나리오 B: 현실적인 채널
    config_impaired = {
        'use_cfo': True, 
        'max_cfo_bins': 0.35, 
        'use_multipath': True,
        'multipath_taps': [1.0, 0.5j, 0.3],
        'multipath_delays': [0, 3, 7]
    }
    evaluate_model(best_trained_model, sim, test_snrs, config_impaired, "Scenario B - Impaired Channel")
    
    print("\n========== [작업 완료] ==========")