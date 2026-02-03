# CallCops: 실시간 오디오 워터마킹 신경망을 이용한 전화 통화 인증 시스템

## 1. 연구 배경 및 문제 정의

### 1.1 보이스피싱 문제

보이스피싱은 발신자의 신원을 위조하여 피해자를 속이는 범죄이다. 기존의 발신번호 표시(Caller ID)는 쉽게 위조 가능하며, 통화 중 상대방이 실제로 주장하는 기관의 직원인지 검증할 수 있는 기술적 수단이 부재하다.

### 1.2 해결 아이디어: 오디오 워터마킹

CallCops는 **오디오 워터마킹(Audio Watermarking)** 기술을 활용하여 이 문제를 해결한다. 핵심 개념은 다음과 같다:

- **발신 측**: 통화 음성에 **사람이 인지할 수 없는** 128비트 인증 정보를 실시간으로 삽입
- **수신 측**: 수신된 음성에서 인증 정보를 추출하여 발신자의 진위를 검증

이 과정에서 워터마크는 다음 조건을 만족해야 한다:

| 조건 | 설명 | 목표 |
|------|------|------|
| 비인지성 (Imperceptibility) | 사람이 들어도 원본과 구별 불가 | PESQ >= 4.0, SNR > 30dB |
| 강건성 (Robustness) | 전화망 코덱 압축을 거쳐도 복원 가능 | BER < 5% (G.729 통과 후) |
| 실시간성 (Real-time) | 40ms 프레임 단위 처리, 지연 최소화 | Latency < 200ms |

### 1.3 디지털 오디오의 기초

#### 1.3.1 아날로그에서 디지털로: 샘플링(Sampling)

소리는 공기의 압력 변화, 즉 **연속적인 아날로그 파형**이다. 이를 컴퓨터에서 처리하려면 디지털로 변환해야 한다. 이 과정이 **샘플링(Sampling)** 이다.

샘플링은 연속 신호를 **일정 시간 간격으로 측정하여 숫자의 나열로 변환**하는 것이다:

```text
아날로그 파형 (연속):  ~~~∿∿~~~∿∿∿~~~
                       ↓ 일정 간격으로 측정
디지털 신호 (이산):    [0.1, 0.3, 0.7, 0.9, 0.6, 0.2, -0.1, -0.4, ...]
```

**샘플링 레이트(Sampling Rate)** 는 1초에 몇 번 측정하는지를 나타낸다:
- 8,000 Hz = 초당 8,000번 측정 → 1초 오디오 = 8,000개의 숫자
- 44,100 Hz = CD 품질 (음악용)
- 48,000 Hz = 방송/영상 표준

#### 1.3.2 나이퀴스트 정리와 8kHz의 의미

**나이퀴스트-섀넌 샘플링 정리(Nyquist-Shannon Sampling Theorem)** 에 따르면, 원본 신호를 완벽히 복원하려면 **신호의 최대 주파수의 2배 이상**으로 샘플링해야 한다:

$$
f_s \geq 2 \cdot f_{\max}
$$

전화망의 주파수 대역은 **300 ~ 3,400 Hz**이다 (인간 음성의 핵심 대역). 최대 주파수 3,400 Hz를 복원하려면:

$$
f_s \geq 2 \times 3{,}400 = 6{,}800 \text{ Hz}
$$

실제로는 여유를 두어 **8,000 Hz**로 샘플링한다. 이 8,000 Hz가 전화망의 국제 표준(ITU-T G.711)이다.

반대로 말하면, 8kHz로 샘플링된 오디오가 표현할 수 있는 최대 주파수는:

$$
f_{\max} = \frac{f_s}{2} = \frac{8{,}000}{2} = 4{,}000 \text{ Hz (나이퀴스트 주파수)}
$$

따라서 CallCops의 워터마크는 0 ~ 4,000 Hz 범위 안에서만 작동한다.

#### 1.3.3 양자화(Quantization)와 비트 깊이

샘플링된 각 값을 저장하려면 **유한한 비트 수로 표현**해야 한다. 이것이 양자화이다.

- **16-bit PCM**: 각 샘플을 -32,768 ~ +32,767 범위의 정수로 표현 ($2^{16} = 65{,}536$ 단계)
- 신경망 학습 시에는 이를 **$[-1.0, +1.0]$ 범위의 실수(float)** 로 정규화하여 사용한다

```text
원본 16-bit:  [-32768, ..., -1, 0, 1, ..., 32767]  (정수)
정규화 후:    [-1.0,   ..., 0.0, ...,        1.0]   (실수)
```

#### 1.3.4 텐서 형태: [B, 1, T]의 의미

신경망에 오디오를 입력할 때 **3차원 텐서(Tensor)** 형태를 사용한다:

```text
Audio Tensor: [B, C, T]
               │  │  │
               │  │  └─ T (Time): 시간 축의 샘플 수
               │  │     예: T=8000이면 1초 (@ 8kHz), T=320이면 40ms
               │  │
               │  └──── C (Channel): 오디오 채널 수
               │        Mono = 1, Stereo = 2
               │        전화 음성은 항상 Mono이므로 C=1
               │
               └─────── B (Batch): 동시에 처리하는 샘플 수
                        학습 시 B=24 (한 번에 24개의 오디오를 동시 처리)
                        추론 시 B=1 (하나씩 처리)
```

**왜 3차원인가?**

PyTorch의 `Conv1d` 연산은 입력으로 `[B, C_in, Length]` 형태를 기대한다. 여기서:

- `B`(배치 차원)는 **GPU 병렬 처리** 를 위해 필수적이다. 24개의 오디오를 하나씩 처리하면 GPU 활용률이 낮지만, 배치로 묶어서 한 번에 처리하면 연산 효율이 극대화된다.
- `C`(채널 차원)는 합성곱의 **입출력 채널**을 나타낸다. 원본 오디오는 C=1(mono)이지만, Conv1d를 거치면 C=32, 64, 128, 256 등 다수의 특징 맵(feature map)으로 확장된다.
- `T`(시간 차원)는 실제 신호의 **시계열 데이터**이다. Conv1d 커널이 이 축을 따라 슬라이딩한다.

**왜 Conv1d(1차원 합성곱)인가?**

이미지는 2차원(높이 × 너비) 신호이므로 Conv2d를 사용하지만, 오디오는 **시간 축 하나만 가진 1차원 신호**이므로 Conv1d를 사용한다. 스펙트로그램(시간 × 주파수)으로 변환하면 Conv2d를 사용할 수도 있지만, CallCops는 **원시 파형(raw waveform)** 을 직접 처리하여 변환 과정의 정보 손실을 피한다.

#### 1.3.5 구체적 수치 예시

1초 분량의 8kHz 모노 오디오를 배치 크기 24로 처리하는 경우:

```text
입력 텐서: [24, 1, 8000]
            │   │    │
            │   │    └─ 8000 samples = 1초 × 8000 Hz
            │   └────── 1 channel (Mono)
            └────────── 24개의 오디오 (한 배치)

메모리 크기: 24 × 1 × 8000 × 4 bytes (float32) = 768 KB
```

5.12초(128 프레임, 1 사이클) 분량:

```text
입력 텐서: [24, 1, 40960]
                      │
                      └─ 40,960 samples = 128 × 320 = 5.12초 × 8000 Hz

메모리 크기: 24 × 1 × 40960 × 4 bytes = 3.93 MB
```

### 1.4 오디오 규격 요약

| 항목 | 값 | 근거 |
|------|-----|------|
| 샘플링 레이트 | 8,000 Hz | ITU-T G.711 전화망 표준 |
| 나이퀴스트 주파수 | 4,000 Hz | $f_s / 2$ |
| 유효 대역폭 | 300 ~ 3,400 Hz | 전화 음성 대역 |
| 비트 깊이 | 16-bit PCM | 65,536 단계 양자화 |
| 채널 | Mono (1채널) | 전화 통화는 단채널 |
| 텐서 형태 | [B, 1, T] | 배치 × 채널 × 시간 |
| 1초 샘플 수 | 8,000 samples | $f_s \times 1\text{s}$ |
| 40ms 프레임 | 320 samples | $8000 \times 0.04$ |

---

## 2. 핵심 설계: Frame-Wise Cyclic Watermarking

### 2.1 프레임 단위 비트 삽입

CallCops의 핵심 아이디어는 **40ms 오디오 프레임 하나에 정확히 1비트**를 삽입하는 것이다.

```
프레임 크기 = 40ms = 320 samples (@ 8kHz)
페이로드    = 128비트 순환 메시지
사이클 길이 = 128 프레임 × 40ms = 5.12초
```

각 프레임 `i`는 메시지의 `i mod 128` 번째 비트를 담당한다:

```
Frame 0   → message[0]
Frame 1   → message[1]
...
Frame 127 → message[127]
Frame 128 → message[0]    ← 다시 처음부터 반복
Frame 129 → message[1]
...
```

### 2.2 왜 순환(Cyclic) 구조인가?

순환 구조를 사용하면 **통화의 임의 지점에서 5.12초만 확보하면** 128비트 전체를 복원할 수 있다. 발신자가 1분을 통화했다면, 수신자는 그 중 아무 5.12초를 선택해도 인증이 가능하다. 5.12초보다 긴 구간을 확보하면 동일 비트가 여러 번 반복되므로, 이들을 **평균(Averaging)** 하여 더 높은 정확도를 얻을 수 있다.

### 2.3 128비트 페이로드 구조

```
[16-bit Sync Pattern][32-bit Timestamp][64-bit Auth Data][16-bit CRC]
       동기화 패턴         타임스탬프       인증 데이터        오류 검출
```

- **Sync Pattern** `1010101010101010`: 프레임 경계 동기화에 사용
- **Timestamp**: 통화 시작 시각 (위변조 방지)
- **Auth Data**: 발신자 인증 정보
- **CRC-16**: 복원된 비트의 무결성 검증

---

## 3. 모델 아키텍처

CallCops는 세 개의 신경망으로 구성된다:

1. **Encoder** (인코더): 원본 오디오 + 메시지 → 워터마크된 오디오
2. **Decoder** (디코더): 워터마크된 오디오 → 추출된 비트
3. **Discriminator** (판별기): 원본 vs 워터마크 오디오 구별 (GAN 학습용)

### 3.1 기본 구성 요소 (Building Blocks)

#### 3.1.1 ConvBlock

1차원 합성곱(Conv1d)을 기반으로 한 기본 블록이다:

```text
입력 [B, C_in, T] → Conv1d → BatchNorm1d → LeakyReLU(0.2) → 출력 [B, C_out, T]
```

각 구성 요소의 역할:

- **Conv1d (1차원 합성곱)**: 학습 가능한 필터(커널)가 시간 축을 따라 슬라이딩하며 패턴을 탐지한다. 예를 들어 커널 크기 7, 입력 채널 1, 출력 채널 32인 Conv1d는 **7개 연속 샘플의 패턴**을 인식하는 32종류의 필터를 학습한다. 각 필터가 서로 다른 오디오 특징(기본 주파수, 에너지 변화, 배음 구조 등)을 포착하게 된다.

  ```text
  입력: [B, 1, 8000]  (1채널, 8000 samples)
          │
    Conv1d(1→32, kernel=7, padding=3)
          │
  출력: [B, 32, 8000]  (32개 특징 맵, 시간 해상도 유지)
  ```

  `padding=3`은 커널 크기 7의 중앙 정렬을 위한 것이다: $(7-1)/2 = 3$. 이로써 입력과 출력의 시간 길이가 동일하게 유지된다.

- **BatchNorm1d (배치 정규화)**: 각 채널의 출력 분포를 평균=0, 분산=1로 정규화한다. 이를 통해 레이어가 깊어져도 값의 분포가 안정적으로 유지되어 학습이 원활해진다.

- **LeakyReLU(0.2) (활성화 함수)**: 비선형 변환을 적용한다. 양수 입력은 그대로, 음수 입력에는 0.2를 곱한다:

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ 0.2x & x \leq 0 \end{cases}
$$

  비선형 활성화 함수가 없으면 아무리 많은 레이어를 쌓아도 하나의 선형 변환과 동일해지므로, 복잡한 패턴을 학습할 수 없다.

전체 블록의 수식:

$$
\text{ConvBlock}(x) = \text{LeakyReLU}\left(\text{BN}\left(\text{Conv1d}(x)\right)\right)
$$

#### 3.1.2 ResidualBlock (잔차 블록)

두 개의 합성곱을 거친 출력에 **입력을 직접 더하는(Skip Connection)** 구조이다:

```
입력 ─────────────────────────────┐
  │                                │
  ├→ Conv1d → BN → LeakyReLU      │ (더하기)
  ├→ Conv1d → BN ──────────────→ (+) → LeakyReLU → 출력
```

수식:

$$
\text{ResBlock}(x) = \text{LeakyReLU}\left(x + \text{BN}\left(\text{Conv}\left(\text{LeakyReLU}\left(\text{BN}\left(\text{Conv}(x)\right)\right)\right)\right)\right)
$$

Skip Connection의 장점:
- **Gradient 전파 개선**: 깊은 네트워크에서도 gradient가 입력까지 직접 전달됨
- **항등 사상 학습**: 블록이 "아무것도 안 하는 것"을 기본으로 하고, 필요한 변화만 학습

추가로 **Dilated Convolution**을 사용한다. Dilation rate를 `1, 2, 4, 8`로 증가시키면, 파라미터 수를 늘리지 않고도 더 넓은 시간적 맥락(Receptive Field)을 포착할 수 있다.

```
Dilation = 1: [x x x]         → 3 samples 커버
Dilation = 2: [x . x . x]     → 5 samples 커버
Dilation = 4: [x . . . x . . . x] → 9 samples 커버
```

#### 3.1.3 SEBlock (Squeeze-and-Excitation)

**채널 주의 메커니즘(Channel Attention)**이다. 어떤 채널이 현재 입력에 더 중요한지를 자동으로 학습한다.

```
입력 [B, C, T]
  │
  ├→ Global Average Pooling → [B, C, 1]     (Squeeze: 시간 축 압축)
  ├→ FC(C → C/8) → LeakyReLU                (차원 축소)
  ├→ FC(C/8 → C) → Sigmoid                  (중요도 가중치 계산)
  │
  └→ 입력 × 가중치 → 출력                    (Excitation: 채널별 재가중)
```

수식:

$$
\text{SE}(x) = x \cdot \sigma\left(W_2 \cdot \text{ReLU}\left(W_1 \cdot \text{GAP}(x)\right)\right)
$$

여기서 GAP는 Global Average Pooling, $\sigma$는 Sigmoid 함수이다.

### 3.2 Encoder (인코더)

인코더는 원본 오디오와 128비트 메시지를 입력받아 **사람이 인지할 수 없는 미세한 섭동(Perturbation)** 을 생성하고, 이를 원본에 더하여 워터마크된 오디오를 출력한다.

#### 전체 처리 흐름

인코더의 입력은 두 가지이다:
- **Audio** `[B, 1, T]`: 원본 오디오. B개의 모노(1채널) 오디오, 각각 T개의 샘플.
- **Message** `[B, 128]`: 삽입할 128비트 메시지. 각 원소는 0.0(비트 0) 또는 1.0(비트 1)의 실수.

```text
입력: Audio [B, 1, T] + Message [B, 128]
                │
                ▼
    ┌──────────────────────────────┐
    │ 1. Audio Feature Extraction  │  4개 ConvBlock (커널 크기 7)
    │    채널: 1 → 32 → 64 → 128 → 256  │  Stride=1 (해상도 유지)
    └──────────┬───────────────────┘
               │ [B, 256, T]
               ▼
    ┌──────────────────────────────┐
    │ 2. FrameWiseFusionBlock      │  프레임별 비트 삽입
    │    (아래 3.2.1에서 상세 설명)  │
    └──────────┬───────────────────┘
               │ [B, 256, T]
               ▼
    ┌──────────────────────────────┐
    │ 3. Post-Fusion Refinement    │  ConvBlock + SEBlock
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │ 4. Residual Blocks (×4)      │  Dilated Conv (1, 2, 4, 8)
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │ 5. Audio Decoder             │  3개 ConvBlock (커널 크기 7)
    │    채널: 256 → 128 → 64 → 32 │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │ 6. Output Layer              │  Conv1d(32 → 1, 커널 7)
    │    perturbation = α × tanh(·)│
    └──────────┬───────────────────┘
               │ [B, 1, T]
               ▼
    Watermarked = Original + Perturbation
    output = clamp(watermarked, -1, 1)
```

**단계별 텐서 변화** (T=8000, 즉 1초 오디오 기준):

| 단계 | 입력 형태 | 출력 형태 | 설명 |
|------|-----------|-----------|------|
| 입력 | - | [B, 1, 8000] | 원본 오디오 (모노, 1초) |
| Feature Extraction | [B, 1, 8000] | [B, 256, 8000] | 1채널 → 256개 특징 맵 추출 |
| FrameWiseFusion | [B, 256, 8000] | [B, 256, 8000] | 비트 정보 융합 (크기 변화 없음) |
| Residual Blocks | [B, 256, 8000] | [B, 256, 8000] | 특징 정제 (크기 변화 없음) |
| Audio Decoder | [B, 256, 8000] | [B, 32, 8000] | 256채널 → 32채널로 축소 |
| Output Layer | [B, 32, 8000] | [B, 1, 8000] | 32채널 → 1채널 (섭동 생성) |
| 최종 출력 | [B, 1, 8000] | [B, 1, 8000] | 원본 + 섭동 = 워터마크 오디오 |

핵심은 **Stride=1**을 유지하여 시간 해상도(T)가 처음부터 끝까지 동일하다는 점이다. 모든 변환은 채널(C) 차원에서만 일어나며, 시간 축의 각 샘플이 1:1로 보존된다. 이 덕분에 40ms 프레임 경계가 정확하게 유지된다.

**채널의 의미**: Feature Extraction 후 256개 채널이 생기는데, 이는 원본 오디오의 **256가지 서로 다른 관점**을 의미한다. 예를 들어 어떤 채널은 저주파 에너지를, 어떤 채널은 고주파 변화를, 또 다른 채널은 음성의 포만트(Formant) 구조를 포착할 수 있다. 이러한 다각적 표현 위에 워터마크 비트를 융합함으로써, 인간이 인지하기 어려운 영역에 정보를 숨길 수 있다.

#### 3.2.1 FrameWiseFusionBlock (프레임별 비트 융합)

이 블록이 CallCops의 핵심이다. 각 40ms 프레임에 해당하는 **정확히 하나의 비트**만 삽입한다.

**처리 과정:**

1. **비트 인덱싱 (Cyclic)**:
   ```
   num_frames = T / 320
   frame_indices = [0, 1, 2, ..., num_frames-1]
   bit_indices = frame_indices mod 128
   frame_bits = message[:, bit_indices]    # 각 프레임의 비트 값 (0 또는 1)
   ```

2. **비트 임베딩**:
   각 비트 값(스칼라 0 or 1)을 256차원 벡터로 변환한다.
   ```
   bit_embed = Linear(1 → 128) → LeakyReLU → Linear(128 → 256)
   ```
   이로써 단일 비트가 풍부한 특징 공간에 매핑된다.

3. **Temporal Modulation**:
   비트 임베딩을 프레임 크기(320 samples)만큼 반복 확장한 후, Depthwise Conv1d로 시간 축 변조를 수행한다.

4. **Gated Fusion**:
   오디오 피처와 비트 피처를 게이트 메커니즘으로 융합한다.

   $$
   \text{gate} = \sigma\left(\text{Conv}_{1\times1}([\text{audio}; \text{bit}])\right)
   $$
   $$
   \text{fused} = \text{gate} \cdot \text{audio} + (1 - \text{gate}) \cdot \text{bit}
   $$

   Gate가 1에 가까우면 원본 오디오를, 0에 가까우면 비트 정보를 더 반영한다. 네트워크는 **비트를 삽입하되 오디오를 최대한 보존하는** 최적의 균형점을 학습한다.

#### 3.2.2 섭동(Perturbation) 생성

인코더의 최종 출력은 `tanh` 활성화를 거친 후 스케일링 팩터 $\alpha$를 곱한다:

$$
\delta = \alpha \cdot \tanh(f_{\theta}(x, m))
$$
$$
\hat{x} = \text{clamp}(x + \delta, -1, 1)
$$

여기서:
- $x$: 원본 오디오
- $m$: 128비트 메시지
- $f_{\theta}$: 인코더 네트워크
- $\alpha$: 학습 가능한 스케일링 팩터, $[0.25, 1.0]$ 범위로 클램프
- $\delta$: 섭동 (워터마크 신호)
- $\hat{x}$: 워터마크된 오디오

`tanh`는 섭동을 $[-1, 1]$ 범위로 제한하고, $\alpha$는 물리적 섭동 에너지의 상한을 결정한다. $\alpha = 0.25$이면 섭동의 최대 진폭이 원본의 25%로 제한된다.

최종적으로 `clamp(watermarked, -1, 1)`로 디지털 오디오의 유효 범위를 벗어나지 않도록 보장한다. 이 범위를 초과하면 **클리핑(Clipping)** 이 발생하여 소리가 찢어지는 왜곡이 생기기 때문이다.

**정리하면**, 인코더의 출력은 원본 오디오와 **동일한 형태 `[B, 1, T]`** 이다. 각 샘플에 $[-0.25, +0.25]$ 범위의 미세한 값이 더해진 것이 전부이므로, 사람의 귀에는 차이가 거의 들리지 않는다.

### 3.3 Decoder (디코더)

디코더는 워터마크된 오디오(또는 코덱을 거친 열화된 오디오)에서 프레임별 비트를 추출한다.

#### 전체 처리 흐름

```
입력: Watermarked Audio [B, 1, T]
              │
              ▼
    ┌─────────────────────────────────┐
    │ 1. Feature Extractor            │  4개 ConvBlock (커널 5, stride 2)
    │    채널: 1 → 32 → 64 → 128 → 256│  시간 해상도: T → T/16
    └──────────┬──────────────────────┘
               │ [B, 256, T/16]
               ▼
    ┌─────────────────────────────────┐
    │ 2. Residual Blocks (×4)         │  Dilated Conv (1, 2, 4, 8)
    └──────────┬──────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │ 3. SE Block                     │  채널 주의 메커니즘
    └──────────┬──────────────────────┘
               │
               ▼
    ┌─────────────────────────────────┐
    │ 4. FrameWiseBitExtractor        │  프레임당 1비트 추출
    │    Conv1d(stride=20, kernel=20) │  (320/16 = 20 다운샘플된 프레임 크기)
    └──────────┬──────────────────────┘
               │ [B, num_frames]
               ▼
    출력: 각 프레임의 비트 로짓 (logit)
```

**단계별 텐서 변화** (T=8000, 즉 1초 오디오 = 25 프레임):

| 단계 | 입력 형태 | 출력 형태 | 설명 |
|------|-----------|-----------|------|
| 입력 | - | [B, 1, 8000] | 워터마크된 오디오 |
| Feature Extraction | [B, 1, 8000] | [B, 256, 500] | stride 2×4 = 16배 다운샘플링 |
| Residual Blocks | [B, 256, 500] | [B, 256, 500] | 특징 정제 |
| SE Block | [B, 256, 500] | [B, 256, 500] | 채널 주의 |
| BitExtractor | [B, 256, 500] | [B, 25] | 프레임당 1비트 추출 |

인코더와 달리 디코더는 **stride=2 합성곱**을 4번 적용하여 시간 해상도를 $2^4 = 16$배 축소한다. 8000 samples → 500 features가 된다. 이렇게 축소하는 이유는 비트 추출에는 세밀한 시간 해상도가 불필요하기 때문이다. 40ms 프레임(320 samples)은 축소 후 $320/16 = 20$ features로 표현되며, 이 20개의 값에서 1비트를 판별하면 충분하다.

#### 3.3.1 FrameWiseBitExtractor

Feature Extractor에서 시간 해상도가 16배 축소되었으므로, 원래 320 samples인 프레임은 320/16 = 20 features로 표현된다.

**처리 과정:**

1. **Feature Refinement**: Depthwise Conv + Pointwise Conv로 피처 정제
2. **Frame-wise Classification**: `Conv1d(kernel=20, stride=20)` 하나로 각 프레임을 **하나의 스칼라 로짓**으로 변환

이 방식의 장점은 **O(1) 복잡도**이다. 입력 길이에 관계없이 프레임당 고정 연산만 수행한다.

#### 3.3.2 128비트 복원 (Cyclic Aggregation)

프레임별 로짓을 128비트 페이로드로 복원하는 과정:

$$
\text{bits}[j] = \frac{1}{|\{i : i \bmod 128 = j\}|} \sum_{i \bmod 128 = j} \text{logit}[i]
$$

즉, 동일한 비트 인덱스를 가진 프레임들의 로짓을 **평균**한다. 10초 오디오(250 프레임)라면, 각 비트는 약 2번의 관측값을 가지므로 노이즈가 줄어든다.

### 3.4 Discriminator (판별기)

Multi-Scale Discriminator를 사용한다. 세 가지 스케일(1x, 2x, 4x 다운샘플)에서 독립적으로 판별하여, 서로 다른 시간 해상도에서의 자연스러움을 평가한다.

```
입력 Audio [B, 1, T]
     │
     ├── Scale 1 (원본)    → Disc_1 → 판별 점수
     ├── Scale 2 (AvgPool 2x) → Disc_2 → 판별 점수
     └── Scale 4 (AvgPool 4x) → Disc_3 → 판별 점수
```

각 판별기는 4개의 Conv1d(stride=2) + BN + LeakyReLU 레이어 후 1차원 출력을 생성한다.

---

## 4. 손실 함수 (Loss Function)

CallCops의 학습은 다수의 손실 함수를 가중 합산한 **복합 손실**을 최소화한다:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{bit}} \cdot \mathcal{L}_{\text{BCE}} + \lambda_{\text{audio}} \cdot \mathcal{L}_{\text{Mel}} + \lambda_{\text{stft}} \cdot \mathcal{L}_{\text{STFT}} + \lambda_{\text{l1}} \cdot \mathcal{L}_{\text{L1}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{Adv}} + \lambda_{\text{det}} \cdot \mathcal{L}_{\text{Det}}
$$

### 4.1 비트 정확도 손실 ($\mathcal{L}_{\text{BCE}}$)

추출된 비트와 원본 비트 간의 **Binary Cross-Entropy**:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left[b_i \log(\hat{b}_i) + (1 - b_i)\log(1 - \hat{b}_i)\right]
$$

여기서 $b_i$는 실제 비트, $\hat{b}_i = \sigma(\text{logit}_i)$는 예측된 비트 확률이다.

수치적 안정성을 위해 `BCEWithLogitsLoss`를 사용한다. 이는 Sigmoid와 BCE를 하나의 연산으로 합쳐 log(0) 문제를 방지한다.

### 4.2 Mel 스펙트로그램 손실 ($\mathcal{L}_{\text{Mel}}$)

인간 청각 특성을 반영한 **다중 해상도 Mel 스펙트로그램** 비교:

$$
\mathcal{L}_{\text{Mel}} = \frac{1}{R} \sum_{r=1}^{R} \left\| \log(\text{Mel}_r(\hat{x}) + \epsilon) - \log(\text{Mel}_r(x) + \epsilon) \right\|_1
$$

여기서:
- $R = 3$: 세 가지 FFT 해상도 (64, 128, 256)
- $\text{Mel}_r$: Mel 필터뱅크를 적용한 스펙트로그램
- $\epsilon = 10^{-7}$: 로그 안정화 상수

**Mel 스케일**은 인간의 주파수 지각 특성을 반영한다:

$$
\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)
$$

낮은 주파수에서는 세밀하게, 높은 주파수에서는 넓게 필터링하여 **인간이 민감하게 느끼는 변화에 더 큰 페널티**를 부과한다.

**8kHz 최적화**: Nyquist 주파수가 4,000Hz이므로, 40개의 Mel 필터뱅크를 80~3,800Hz 범위에 배치한다.

### 4.3 Multi-Resolution STFT 손실 ($\mathcal{L}_{\text{STFT}}$)

**Spectral Convergence Loss**와 **Log Magnitude Loss**의 합:

$$
\mathcal{L}_{\text{SC}} = \frac{\|S_x - S_{\hat{x}}\|_F}{\|S_x\|_F}
$$
$$
\mathcal{L}_{\text{Mag}} = \|\log S_x - \log S_{\hat{x}}\|_1
$$
$$
\mathcal{L}_{\text{STFT}} = \frac{1}{R}\sum_{r=1}^{R}\left(\mathcal{L}_{\text{SC}}^{(r)} + \mathcal{L}_{\text{Mag}}^{(r)}\right)
$$

$\|\cdot\|_F$는 프로베니우스 노름(행렬의 모든 원소 제곱합의 제곱근)이다.

**STFT(Short-Time Fourier Transform)** 는 오디오를 짧은 윈도우로 잘라서 각 구간의 주파수 성분을 분석하는 기법이다. 이를 통해 "시간에 따라 주파수가 어떻게 변하는지"를 2차원 표현(스펙트로그램)으로 얻는다.

**왜 다중 해상도(Multi-Resolution)인가?** STFT에는 시간 해상도와 주파수 해상도 사이에 트레이드오프가 존재한다:
- **작은 윈도우** (64 samples = 8ms): 시간 해상도가 높지만 주파수 해상도가 낮음 → 빠른 변화 감지에 유리
- **큰 윈도우** (256 samples = 32ms): 주파수 해상도가 높지만 시간 해상도가 낮음 → 음조/배음 분석에 유리

세 가지 크기(64, 128, 256)에서 모두 계산하여 평균함으로써, **시간적 왜곡과 주파수적 왜곡을 동시에 포착**한다.

### 4.4 L1 파형 손실 ($\mathcal{L}_{\text{L1}}$)

원본과 워터마크 파형 간의 직접적인 L1 거리:

$$
\mathcal{L}_{\text{L1}} = \frac{1}{T}\sum_{t=1}^{T}|x_t - \hat{x}_t|
$$

이 손실은 SNR(Signal-to-Noise Ratio)을 직접 개선한다. Mel 손실이 지각적 품질에 초점을 맞추는 반면, L1 손실은 파형 수준의 왜곡을 최소화한다.

### 4.5 적대적 손실 ($\mathcal{L}_{\text{Adv}}$)

**LSGAN (Least Squares GAN)** 방식을 사용한다:

$$
\mathcal{L}_{\text{Adv}}^{(G)} = \frac{1}{K}\sum_{k=1}^{K}\mathbb{E}\left[(D_k(\hat{x}) - 1)^2\right]
$$
$$
\mathcal{L}_{\text{Adv}}^{(D)} = \frac{1}{2K}\sum_{k=1}^{K}\left[\mathbb{E}\left[(D_k(x) - 1)^2\right] + \mathbb{E}\left[D_k(\hat{x})^2\right]\right]
$$

$K = 3$은 Multi-Scale Discriminator의 스케일 수이다. Generator는 판별기를 속여 "진짜"로 판단하게 하고, Discriminator는 원본과 워터마크를 구별하도록 학습한다.

LSGAN은 vanilla GAN보다 학습이 안정적이며, mode collapse가 적다.

### 4.6 탐지 손실 ($\mathcal{L}_{\text{Det}}$)

워터마크의 존재 자체를 탐지하는 이진 분류 손실:

$$
\mathcal{L}_{\text{Det}} = \text{BCE}(\text{detection\_logit}, 1)
$$

디코더가 "이 오디오에 워터마크가 있다"고 확신할 수 있도록 학습한다.

### 4.7 손실 가중치 설정

| 가중치 | 기본값 | 역할 |
|--------|--------|------|
| $\lambda_{\text{bit}}$ | 2.0 | 비트 정확도 압박 강도 |
| $\lambda_{\text{audio}}$ | 50.0 | 음질 보존 압박 강도 |
| $\lambda_{\text{stft}}$ | 10.0 | 스펙트럼 보존 |
| $\lambda_{\text{l1}}$ | 1.0 | 파형 수준 SNR |
| $\lambda_{\text{adv}}$ | 0.05 | 자연스러움 |
| $\lambda_{\text{det}}$ | 0.1 | 탐지 확실성 |

**설계 원칙**: $\lambda_{\text{audio}} \gg \lambda_{\text{bit}}$으로 설정하여 **음질을 최우선**으로 보존한다. 비트 정확도는 BCH(511, 128) 오류 정정 코드가 최대 10%의 BER을 허용하므로, 완벽하지 않아도 된다.

---

## 5. 코덱 시뮬레이터 (Codec Simulator)

### 5.1 필요성

전화 통화 음성은 발신에서 수신까지 다양한 코덱을 거친다. 워터마크가 이 과정을 거쳐도 살아남으려면, 학습 중에 코덱의 효과를 시뮬레이션해야 한다.

### 5.2 미분 가능한 코덱 근사

코덱의 양자화(Quantization)는 미분 불가능한 연산이다. 이를 학습에 사용하기 위해 **Straight-Through Estimator (STE)** 를 적용한다:

- **순전파 (Forward)**: 실제 양자화를 수행 ($\text{round}$ 연산)
- **역전파 (Backward)**: Gradient를 그대로 통과시킴 (Identity gradient)

$$
\text{Forward: } Q(x) = \text{round}\left(\frac{x + 1}{2} \cdot (L - 1)\right) \cdot \frac{2}{L-1} - 1
$$
$$
\text{Backward: } \frac{\partial Q}{\partial x} \approx 1 \quad \text{(STE)}
$$

여기서 $L$은 양자화 레벨 수 (8-bit: $L = 256$).

### 5.3 지원하는 코덱

#### G.711 (64kbps)

PCM 코덱으로, **압신(Companding)** 후 8-bit 양자화를 수행한다.

**$\mu$-law 압신** (북미 표준, $\mu = 255$):

$$
F(x) = \text{sign}(x) \cdot \frac{\ln(1 + \mu|x|)}{\ln(1 + \mu)}
$$

**A-law 압신** (한국/유럽 표준, $A = 87.6$):

$$
F(x) = \begin{cases}
\frac{A|x|}{1 + \ln A} & |x| < \frac{1}{A} \\
\frac{1 + \ln(A|x|)}{1 + \ln A} & |x| \geq \frac{1}{A}
\end{cases}
$$

압신은 작은 진폭 신호의 SNR을 개선하여 동적 범위를 효과적으로 압축한다.

#### G.729 (8kbps, 근사)

CELP(Code-Excited Linear Prediction) 기반 고압축 코덱의 **근사 시뮬레이션**:

1. **Low-pass Filtering**: 고주파 정보 손실 모사
2. **Frame-based Quantization**: 10ms 프레임 단위 에너지 정규화 + 양자화 노이즈
3. **Spectral Smoothing**: CELP의 스펙트럼 평활화 효과

#### AMR-NB (4.75~12.2 kbps)

VoLTE/3G에서 사용하는 **ACELP** 기반 코덱의 근사:

- 8가지 비트레이트 모드 (MR475 ~ MR122)
- 학습 중 **랜덤 비트레이트 선택**으로 다양한 네트워크 환경에 대한 강건성 확보
- 20ms 프레임 단위 처리

### 5.4 통합 코덱 시뮬레이터

학습 중 각 배치에서 코덱을 **확률적으로 선택**한다:

```
P(G.711 A-law) = 0.2,  P(G.711 μ-law) = 0.2
P(G.729) = 0.2,        P(AMR-NB) = 0.2
P(코덱 없음) = 0.2
```

추가로 **Curriculum Learning**을 적용하여, 학습 초기에는 "코덱 없음"의 비율을 높이고 점차 어려운 코덱의 비율을 증가시킨다.

---

## 6. 학습 과정

### 6.1 End-to-End 학습의 원리

CallCops의 학습은 **End-to-End(종단 간)** 방식이다. 인코더와 디코더를 따로 학습하지 않고, 전체 파이프라인을 **한 번에** 학습한다:

```text
원본 오디오 x ──→ Encoder ──→ 워터마크 x̂ ──→ [코덱 시뮬레이션] ──→ Decoder ──→ 추출된 비트 b̂
       │              │              │                                              │
       │          message m      perturbation δ                                 target bits m
       │                              │                                              │
       └──── 음질 손실 비교 ────────┘                                              │
                                                                                     │
                                      └────────────── 비트 정확도 손실 비교 ────────┘
```

역전파(Backpropagation)를 통해 gradient가 디코더 → 코덱 시뮬레이터(STE) → 인코더 순서로 전달된다. 인코더는 "디코더가 정확히 복원할 수 있으면서도, 원본과 최대한 비슷한" 섭동을 생성하도록 학습된다.

### 6.2 GAN 학습 루프

**GAN(Generative Adversarial Network)** 은 두 네트워크가 서로 경쟁하며 학습하는 프레임워크이다:
- **Generator** (인코더+디코더): "판별기를 속일 만큼 자연스러운" 워터마크 오디오를 생성
- **Discriminator** (판별기): 원본과 워터마크 오디오를 "구별"하려고 시도

이 경쟁 관계가 수렴하면, 판별기조차 구분할 수 없을 만큼 자연스러운 워터마크가 생성된다.

각 학습 스텝은 두 단계로 진행된다:

**Step 1: Discriminator 업데이트**

```
1. 원본 오디오 → Discriminator → "진짜" 판별
2. 워터마크 오디오 → Discriminator → "가짜" 판별
3. D_loss = LSGAN_loss(real_pred, fake_pred)
4. Discriminator 파라미터 업데이트
```

**Step 2: Generator (Encoder + Decoder) 업데이트**

```
1. Encoder(audio, message) → watermarked
2. CodecSimulator(watermarked) → degraded  (확률적)
3. Decoder(degraded) → predicted_logits
4. Discriminator(watermarked) → fool_score
5. G_loss = λ_bit·L_BCE + λ_audio·L_Mel + λ_stft·L_STFT
            + λ_l1·L_L1 + λ_adv·L_Adv + λ_det·L_Det
6. Encoder/Decoder 파라미터 업데이트
```

### 6.2 학습 전략

| 항목 | 설정 |
|------|------|
| Optimizer | Adam ($\beta_1 = 0.5$, $\beta_2 = 0.9$) |
| Learning Rate | $1 \times 10^{-5}$ |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient Clipping | Max norm = 1.0 |
| Mixed Precision | BFloat16 (AMP) |
| Batch Size | 24 |
| Epochs | 100 |

### 6.3 동적 가중치 제어 (Dynamic Weight Controller)

학습 중 BER과 SNR을 모니터링하여 가중치를 **자동 조정**한다:

- **BER < 2%** (비트 과최적화): $\lambda_{\text{bit}} \leftarrow \max(0.1, \lambda_{\text{bit}} \times 0.8)$
- **SNR < 15dB** (음질 저하): $\lambda_{\text{audio}} \leftarrow \min(500, \lambda_{\text{audio}} \times 1.1)$

이를 통해 **파괴적 수렴(Destructive Convergence)** — 비트 정확도를 위해 음질을 희생하는 현상 — 을 방지한다.

### 6.4 조기 종료 (Early Stopping)

Val SNR이 설정된 임계값(15dB) 미만으로 연속 N 에포크(patience=10) 유지되면 학습을 자동 종료한다. 이는 모델이 "음질 파괴" 방향으로 학습하는 것을 조기에 차단한다.

---

## 7. 데이터 파이프라인

### 7.1 학습 데이터

**AI Hub 저음질 전화망 음성인식 데이터셋 (Dataset 571)** 을 사용한다.

- 한국어 상담 통화 음성 (8kHz)
- 화자 유형: 상담사(Counselor), 고객(Customer)
- JSON 메타데이터 포함 (전사 텍스트, 화자 정보, 발화 시간 등)

### 7.2 데이터 전처리

1. **리샘플링**: 16kHz 원본 → 8kHz 다운샘플링
2. **정규화**: 최대 진폭 기준 $[-1, 1]$ 범위로 정규화
3. **최소 길이 필터링**: 3초 미만 오디오 제외
4. **세그먼테이션**: 긴 오디오를 5.12초(128 프레임) 단위로 분할

### 7.3 데이터 증강

전화망 환경을 시뮬레이션하는 실시간 증강:

| 증강 기법 | 확률 | 설명 |
|-----------|------|------|
| 대역 통과 필터 | 50% | 300~3,400Hz (Sinc 필터 기반) |
| 가우시안 노이즈 | 50% | SNR 15~40dB |
| 음량 변화 | 30% | ±6dB |
| 클리핑 | 10% | 과부하 시뮬레이션 (임계값 0.7~0.95) |

---

## 8. 두 가지 아키텍처

### 8.1 Non-Causal (기본 모델)

- **패딩**: 양쪽 대칭 (symmetric padding)
- **정규화**: BatchNorm1d
- **특징**: 미래 샘플도 참조하므로 약간 더 높은 품질
- **스트리밍**: 히스토리 버퍼 필요 (StreamingEncoderWrapper)

### 8.2 Causal (실시간 모델)

- **패딩**: 왼쪽만 (left-only padding)
- **정규화**: InstanceNorm1d (스트리밍 안정성)
- **특징**: 미래 참조 없음 (Zero Look-ahead)
- **스트리밍**: 히스토리 버퍼 불필요, 단일 프레임 직접 처리

Causal 모델의 핵심 차이:

```python
# Non-Causal: 양쪽 패딩
padding = (kernel_size - 1) * dilation // 2

# Causal: 왼쪽만 패딩
padding = (kernel_size - 1) * dilation  # 전부 왼쪽에
x = F.pad(x, (causal_padding, 0))      # (left, right=0)
```

이로써 출력의 각 샘플이 **오직 과거 입력에만 의존**하게 된다.

**Receptive Field 비교:**

| 구성 요소 | Non-Causal (편측) | Causal (과거만) |
|-----------|------------------|----------------|
| Encoder (4 Conv, k=7) | ~12 samples | 24 samples |
| Residual (4 blocks, dilation 1,2,4,8) | ~15 samples | 30 samples |
| Decoder (3 Conv, k=7) | ~9 samples | 18 samples |
| Output (k=7) | ~3 samples | 6 samples |
| **합계** | **~57 samples (~7ms)** | **~78 samples (~10ms)** |

### 8.3 스트리밍 추론

#### Non-Causal 스트리밍 (StreamingEncoderWrapper)

Non-Causal 모델은 대칭 패딩으로 인해 미래 샘플이 필요하다. 이를 해결하기 위해 **롤링 히스토리 버퍼**를 유지한다:

```
[히스토리 5프레임: 1,600 samples] + [새 프레임: 320 samples] = 1,920 samples
                                                                    │
                                                     ONNX 추론 (encoder)
                                                                    │
                                              출력에서 마지막 320 samples만 추출
```

- 히스토리에는 **원본(RAW) 오디오**를 저장 (워터마크 오디오를 저장하면 이중 삽입 발생)
- 메시지 회전(rotation)으로 글로벌 비트 인덱스 동기화:
  $$\text{offset} = (\text{globalFrameIndex} - \text{historyFilled}) \bmod 128$$

#### Causal 스트리밍 (CausalStreamingEncoder)

Causal 모델은 히스토리 버퍼 없이 320 samples 하나만 입력하면 된다:

```
새 프레임 320 samples → CausalEncoder → 워터마크된 320 samples
```

---

## 9. 웹 데모 (Frontend)

React 19 + Vite 7 기반 웹 애플리케이션으로, 브라우저에서 실시간 워터마킹을 시연한다.

### 9.1 ONNX Runtime Web

PyTorch 모델을 ONNX 형식으로 변환하고 INT8 양자화하여 브라우저에서 실행한다:

- **Encoder INT8**: ~2MB (워터마크 삽입)
- **Decoder INT8**: ~2MB (워터마크 추출)

### 9.2 브라우저 환경에서의 8kHz 변환

브라우저의 Web Audio API는 마이크 입력을 보통 **44,100Hz 또는 48,000Hz**로 캡처한다. 하지만 CallCops의 모델은 8kHz 입력을 기대하므로 **다운샘플링(Downsampling)** 이 필요하다.

이를 위해 `OfflineAudioContext`를 사용한 **Polyphase Interpolation** 리샘플링을 수행한다:

```text
48,000 Hz 마이크 입력
       │
  OfflineAudioContext (48kHz → 8kHz)
  내부적으로 Anti-aliasing LPF + 6배 다운샘플링
       │
  8,000 Hz 오디오 → 모델 입력
```

48kHz → 8kHz 변환 시 나이퀴스트 정리에 의해 4,000Hz 이상의 주파수 성분은 **안티앨리어싱 저역통과 필터(Anti-aliasing LPF)** 로 제거된다. 이는 전화망이 물리적으로 수행하는 대역 제한과 동일한 효과이다.

### 9.3 실시간 스트리밍 흐름

```text
마이크 (48kHz) → ScriptProcessor → 입력 큐 (inputQueue)
                                         │
                                         ▼
                              8kHz 다운샘플링 → pendingBuffer
                                         │
                                         ▼
                              processLoop (순차 처리):
                                While pendingBuffer >= 320 samples:
                                  wrapper.processFrame(frame, message)
                                  → watermarkedChunks에 추가
                                         │
                                         ▼
                              종료 시 → 잔여 프레임 처리 → WAV 생성
```

---

## 10. 평가 지표

### 10.1 SNR (Signal-to-Noise Ratio)

$$
\text{SNR} = 10 \cdot \log_{10}\left(\frac{\mathbb{E}[x^2]}{\mathbb{E}[(x - \hat{x})^2]}\right) \text{ [dB]}
$$

원본 대비 워터마크 잡음의 비율. 높을수록 좋다.

### 10.2 BER (Bit Error Rate)

$$
\text{BER} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat{b}_i \neq b_i]
$$

복원된 비트 중 오류 비율. 낮을수록 좋다. BCH 오류 정정을 적용하면 10%까지 허용 가능하다.

### 10.3 PESQ (Perceptual Evaluation of Speech Quality)

ITU-T P.862 표준 음질 평가 지표. 1.0(최악) ~ 4.5(최상). 목표는 4.0 이상 (사람이 차이를 인지하기 어려운 수준).

---

## 11. 결론 및 향후 과제

### 11.1 기술적 기여

1. **Frame-Wise Cyclic Watermarking**: 40ms 프레임 = 1비트의 정밀한 시간적 매핑으로 임의 지점 탐지 가능
2. **미분 가능한 코덱 시뮬레이터**: STE 기반으로 G.711/G.729/AMR-NB 코덱을 학습 루프에 통합
3. **Causal Architecture**: Zero look-ahead로 실시간 VoIP 삽입 가능
4. **동적 가중치 제어**: BER/SNR 모니터링 기반 자동 학습 균형 조절

### 11.2 향후 과제

- **대규모 데이터셋**: 더 다양한 화자/환경의 음성으로 일반화 성능 향상
- **Noise Robustness**: 실제 전화 환경의 배경 잡음에 대한 강건성 강화
- **모바일 최적화**: React Native 앱을 통한 온디바이스 실시간 추론
- **PESQ 최적화**: 지각적 음질 4.0 이상 달성을 위한 손실 함수 개선

---

## 부록 A: 모델 파라미터 수

| 구성 요소 | 채널 구성 | 파라미터 수 (약) |
|-----------|-----------|-----------------|
| Encoder | [1→32→64→128→256] + Fusion + Residual×4 + [256→128→64→32→1] | ~2.5M |
| Decoder | [1→32→64→128→256] (stride 2) + Residual×4 + SE + BitExtractor | ~1.5M |
| Discriminator | 3-scale × [1→32→64→128→256→1] | ~1.0M |
| **합계** | | **~5.0M** |

## 부록 B: 128비트 페이로드 구조

```
비트 위치    : [0───15] [16──────47] [48──────────111] [112────127]
내용         : Sync     Timestamp    Auth Data         CRC-16
크기         : 16-bit   32-bit       64-bit            16-bit
역할         : 동기화    시각 정보     인증 데이터        무결성 검증
```

## 부록 C: 용어 정리

| 용어 | 설명 |
|------|------|
| Conv1d | 1차원 합성곱 연산. 시간 축을 따라 커널을 슬라이딩하며 특징 추출 |
| BatchNorm | 배치 정규화. 학습 안정성을 위해 각 채널의 출력 분포를 표준화 |
| InstanceNorm | 인스턴스 정규화. 샘플별 독립 정규화로 스트리밍 추론에 적합 |
| LeakyReLU | 활성화 함수. 음수 영역에 작은 기울기를 허용하여 gradient 소실 방지 |
| Residual Connection | 잔차 연결. 입력을 출력에 직접 더하여 gradient 전파 개선 |
| Dilated Convolution | 팽창 합성곱. 커널 원소 사이에 공간을 두어 넓은 수용 영역 확보 |
| STFT | Short-Time Fourier Transform. 시간-주파수 영역 분석 |
| Mel Scale | 인간 청각 특성을 반영한 주파수 척도 |
| GAN | Generative Adversarial Network. 생성기와 판별기의 적대적 학습 |
| LSGAN | Least Squares GAN. MSE 기반 GAN으로 안정적 학습 |
| STE | Straight-Through Estimator. 미분 불가능 연산의 gradient를 근사 |
| BER | Bit Error Rate. 비트 오류율 |
| SNR | Signal-to-Noise Ratio. 신호 대 잡음비 |
| PESQ | Perceptual Evaluation of Speech Quality. 지각적 음질 평가 지표 |
| BCH Code | Bose-Chaudhuri-Hocquenghem 오류 정정 코드 |
| ONNX | Open Neural Network Exchange. 모델 교환 표준 형식 |
