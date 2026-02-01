# CallCops 👮‍♂️📞

> **한국어 전화 통화 인증을 위한 실시간 오디오 워터마킹 시스템**
> Real-time Audio Watermarking System for Korean Phone Call Verification

CallCops는 보이스피싱 방지 및 통화 무결성 인증을 위해 설계된 딥러닝 기반 오디오 워터마킹 솔루션입니다. 
사람이 인지할 수 없는 비가청 워터마크(128-bit)를 실시간으로 음성에 삽입하며, 전화망의 낮은 대역폭(8kHz)과 G.729 코덱 압축 환경에서도 높은 정확도로 검출할 수 있습니다.

## 🚀 주요 기능 (Key Features)

- **🎧 비가청 워터마킹 (Inaudible Watermarking)**: 통화 품질을 저해하지 않으면서(PESQ ≥ 4.0) 디지털 서명을 삽입합니다.
- **⚡ 실시간 처리 (Real-time Processing)**: 200ms 미만의 초저지연설계로 실제 통화 중 실시간 탐지가 가능합니다.
- **🛡️ 코덱 내성 (Codec Robustness)**: G.711/G.729 등 고압축 코덱을 거친 후에도 95% 이상의 검출률을 보장합니다.
- **📱 온디바이스 AI (On-device AI)**: 서버 통신 없이 모바일/웹에서 직접 동작하는 경량화 모델(ONNX Runtime)을 탑재했습니다.
- **🔄 랜덤 위치 탐지**: 워터마크가 반복 삽입되어 통화 도중 언제든 검증이 가능합니다.

## 🛠 사용된 기술 (Tech Stack)

### **Frontend & Mobile**
- **Framework**: React 19, Vite
- **Styling**: TailwindCSS
- **Audio Processing**: Wavesurfer.js, Web Audio API
- **AI Inference**: ONNX Runtime Web (WASM/WebGL)

### **AI Model & Backend**
- **Deep Learning**: PyTorch, Custom CNN/Attention Architecture
- **Optimization**: Quantization (INT8), TorchScript, ONNX Export
- **Audio Codec**: G.711 / G.729 Simulator

## 👥 팀원 (Team Members)

> *아래에 팀원 정보를 입력해주세요*

- **[이름]**: [역할/담당]
- **[이름]**: [역할/담당]
- **[이름]**: [역할/담당]
- **[이름]**: [역할/담당]

## 📸 스크린샷 및 시연 영상 (Demo)

> *아래에 시연용 움짤(GIF) 4개 이상 또는 20초 이상의 동영상을 첨부해주세요.*
> *이미지는 `assets/` 폴더 등에 넣고 링크를 걸면 좋습니다.*

### 1. 워터마크 삽입 (Embedding)
*(여기에 GIF/이미지 추가)*

### 2. 실시간 검출 (Real-time Detection)
*(여기에 GIF/이미지 추가)*

### 3. 코덱 압축 테스트
*(여기에 GIF/이미지 추가)*

### 4. 모바일 구동 화면
*(여기에 GIF/이미지 추가)*

## 📥 설치 및 다운로드 (Download)

### **Android APK 다운로드**
> *아래에 APK 파일 다운로드 링크를 넣어주세요 (Google Drive 링크 등)*

[📱 APK 다운로드 링크](https://drive.google.com/...)

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