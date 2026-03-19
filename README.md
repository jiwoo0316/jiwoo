
# LoRa Demodulation via Phase-Aware 1D CNN
**딥러닝(1D CNN)을 활용한 열악한 채널 환경(CFO + Multipath)에서의 LoRa 신호 복조 시스템**

본 프로젝트는 사물인터넷(IoT)의 핵심 통신 기술인 LoRa 시스템이 실제 도심지나 악조건 채널에서 겪는 스펙트럼 누수 문제를 딥러닝으로 극복하기 위해 설계된 연구용 파이프라인임.

기존의 단순 Dechirp+FFT 기반 클래식 수신기의 수학적 한계를 넘기 위해, 복소 스펙트럼의 위상 정보를 온전히 활용하는 2-Channel 1D CNN 아키텍처를 제안함.

---

## 주요 특징

#### 정밀한 통신 물리계층 시뮬레이터

이산시간 기반의 완벽한 위상 연속 첩(Chirp) 신호 생성.

 AWGN 뿐만 아니라, 현실적인 채널 장애물인 다중 경로 페이딩 및 반송파 주파수 오프셋 완벽 모사.

공정한 클래식 베이스라인 탑재하여 정확한 OSR 보정이 적용된 Naive FFT 수신기.

CFO 환경에 대응하기 위해 인접 주파수 에너지를 합산하는 고급 Grouped-Bin 수신기 구현.

Phase-Aware 1D CNN 아키텍처를 사용한 절댓값 연산으로 위상 정보가 유실되는 기존 연구의 한계를 극복.

Dechirp+FFT의 결과를 실수부와 허수부 2채널로 분리 입력하여 비선형적 위상 왜곡 패턴 학습.

#### 검증 루프
Validation Loss 기반의 Best Model Checkpointing.

Low-SER 구간의 통계적 신뢰도 확보를 위해 SNR 당 최대 50,000회의 대규모 몬테카를로 시뮬레이션 동적 수행.

---

## 개발 환경 및 실행방법

이 코드는 Python 3.13.2 환경에서 작성되었고, 다음 라이브러리가 필요함.

`numpy`
`scipy`
`matplotlib`
`torch` (PyTorch)

```bash
# 라이브러리 설치
pip install numpy scipy matplotlib torch
```

```bash
# 실행방법
python main.py
```


