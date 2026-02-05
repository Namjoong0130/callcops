# CallCops 👮‍♂️📞

> **한국어 전화 통화 인증을 위한 실시간 오디오 워터마킹 시스템**
> Real-time Audio Watermarking System for Korean Phone Call Verification

CallCops는 보이스피싱 방지 및 통화 무결성 인증을 위해 설계된 딥러닝 기반 오디오 워터마킹 솔루션입니다. 
사람이 인지할 수 없는 비가청 워터마크(128-bit)를 실시간으로 음성에 삽입하며, 전화망의 낮은 대역폭(8kHz)과 G.729 코덱 압축 환경에서도 높은 정확도로 검출할 수 있습니다.

## 🚀 주요 기능 (Key Features)

- **🎧 비가청 워터마킹 (Inaudible Watermarking)**: 통화 품질을 저해하지 않으면서(PESQ ≥ 4.0) 디지털 서명을 삽입합니다.
- **⚡ 실시간 처리 (Real-time Processing)**: Causal 신경망과 200ms 미만의 초저지연 설계로 실제 통화 중 실시간 탐지가 가능합니다.
- **🛡️ 강력한 오류 정정 (Error Correction)**: Reed-Solomon RS(16,12) 코드를 탑재하여 심한 노이즈와 코덱 왜곡 환경에서도 데이터 무결성을 보장합니다.
- **📱 온디바이스 AI (On-device AI)**: 서버 통신 없이 모바일/웹에서 직접 동작하는 경량화 모델(ONNX Runtime)을 탑재했습니다.
- **🔄 랜덤 위치 탐지**: 워터마크가 반복 삽입되어 통화 도중 언제든 검증이 가능합니다.

## 🛠 사용된 기술 (Tech Stack)

### **Frontend & Mobile**
- **Framework**: React 19, Vite
- **Styling**: TailwindCSS
- **Audio Processing**: Wavesurfer.js, Web Audio API
- **AI Inference**: ONNX Runtime Web (WASM/WebGL)

### **AI Model & Backend**
- **Deep Learning**: PyTorch, Causal CNN/Attention Architecture
- **Optimization**: Quantization (INT8), TorchScript, ONNX Export
- **Error Correction**: Reed-Solomon (16, 12) over GF(2^8)
- **Audio Codec**: G.711 / G.729 Simulator

## 👥 팀원 (Team Members)

- 안준영: 풀스택
- 임남중: 프론트

## 📸 스크린샷 및 시연 영상 (Demo)


## 📥 설치 및 다운로드 (Download)

### **Android APK 다운로드**

---

## 💻 실행 방법 (How to Run)

### Frontend (Web Demo)
```bash
cd callcops-frontend
npm install
npm run dev
```

### Model Training
```bash
cd callcops-model
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```

## 📄 라이선스 (License)
This project is licensed under the MIT License.
