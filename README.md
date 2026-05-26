# SA-DMAE: Spatial-Axial Denoising Masked Autoencoder

> 컴퓨터비전 수업 프로젝트 — 다반 7팀  
> DMAE (ICLR 2023) 를 2.5D 뇌종양 MRI 복원에 맞게 확장한 모델

---

## 프로젝트 구조

```
SA-DMAE/
├── models_sa_dmae.py       # SA-DMAE 모델 (핵심)
├── dataset_medical.py      # BraTS 2021 / UCSF-PDGM 데이터셋
├── main_pretrain_sa.py     # Pre-training 스크립트
├── colab_preprocess.ipynb  # Colab 전처리 노트북 (3D NIfTI → 2.5D .pt)
├── util/
│   ├── misc.py             # 학습 유틸 (torch 2.x 호환)
│   ├── pos_embed.py        # Positional embedding (numpy 2.x 호환)
│   └── lr_sched.py
└── requirements.txt
```

---

## 아키텍처

```
Input: (B, 3, 3, 224, 224)
       └─ 3장 연속 axial 슬라이스 × 3채널 (T1ce / T2 / FLAIR)
              ↓
  ┌─── Spatial Stream ───┐      ┌─── Axial Stream ────┐
  │  center slice         │      │  prev / center / next│
  │  + Gaussian noise     │      │  (no masking)        │
  │  + random masking     │      │  shared ViT weights  │
  └──────────┬────────────┘      └──────────┬───────────┘
             │                              │
             └──────────┬───────────────────┘
                        ↓
              ViT Block × 8  (shared weights)
                        ↓
         [ViT Block → AxialAttentionBlock] × 4
              Cross-Attention: Spatial(Q) ← Axial(KV)
                        ↓
                  Decoder (DMAE 동일)
                        ↓
              MSE Reconstruction Loss (center slice 기준)
```

| 컴포넌트 | 설명 |
|----------|------|
| **Spatial Stream** | 중앙 슬라이스에 노이즈+마스킹 → 원본 복원 (기존 DMAE와 동일) |
| **Axial Stream** | 인접 슬라이스(prev/next) 문맥 인코딩, Spatial과 ViT 가중치 공유 |
| **AxialAttentionBlock** | Cross-Attention으로 인접 슬라이스 정보를 중앙 슬라이스에 융합 |

---

## 데이터셋

| 데이터셋 | 케이스 수 | 모달리티 | 비고 |
|----------|-----------|----------|------|
| BraTS 2021 | 1,251 | T1ce / T2 / FLAIR | seg 파일로 종양 center 추출 |
| UCSF-PDGM | 318 | T1ce / T2 / FLAIR | T2 없는 183개 스킵 |
| **합계** | **1,569** | | Pre-training에 사용 |

**전처리 방식:** 3D NIfTI → 종양 ROI 중심 Z축 기준 연속 3장 axial 슬라이스 추출 → `(3, 3, 224, 224)` `.pt` 파일

---

## 환경 설정

```bash
git clone https://github.com/whkim4338/SA-DMAE.git
cd SA-DMAE

pip install -r requirements.txt
```

**주요 의존성:** `torch 2.x`, `timm 1.x`, `nibabel`, `numpy 2.x`

---

## 데이터 준비

### Colab 전처리 (권장)

1. BraTS 2021 / UCSF-PDGM 데이터를 Google Drive에 업로드
2. `colab_preprocess.ipynb` 를 Colab에서 실행 — Cell 3의 경로만 수정
3. 생성된 `.pt` 파일을 로컬에 다운로드

```
pt_slices/
├── brats/   ← BraTS .pt 파일 (1,251개)
└── ucsf/    ← UCSF .pt 파일 (318개)
```

### 로컬 NIfTI 직접 사용

```
datasets/BraTS2021/
└── BraTS2021_00000/
    ├── BraTS2021_00000_t1ce.nii.gz
    ├── BraTS2021_00000_t2.nii.gz
    ├── BraTS2021_00000_flair.nii.gz
    └── BraTS2021_00000_seg.nii.gz
```

---

## Pre-training 실행

### BraTS + UCSF `.pt` 파일로 학습 (GPU 권장)

```bash
python main_pretrain_sa.py \
    --data_path  ./pt_slices/brats \
    --ucsf_path  ./pt_slices/ucsf \
    --preprocessed \
    --device cuda \
    --model sa_dmae_vit_base_patch16 \
    --epochs 200 \
    --batch_size 16 \
    --accum_iter 2 \
    --sigma 0.25 \
    --mask_ratio 0.75 \
    --output_dir ./output_sa
```

### NIfTI 직접 학습

```bash
python main_pretrain_sa.py \
    --data_path ./datasets/BraTS2021 \
    --device cuda \
    --model sa_dmae_vit_base_patch16 \
    --epochs 200 \
    --batch_size 8
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `sa_dmae_vit_base_patch16` | 모델 크기 (`base` / `large`) |
| `--n_slices` | `3` | 연속 슬라이스 수 (홀수) |
| `--axial_depth` | `4` | AxialAttentionBlock 삽입 레이어 수 |
| `--sigma` | `0.25` | Gaussian noise 표준편차 |
| `--mask_ratio` | `0.75` | 마스킹 비율 |
| `--preprocessed` | `False` | `.pt` 파일 사용 여부 |
| `--ucsf_path` | `""` | UCSF-PDGM `.pt` 폴더 경로 (`--preprocessed` 와 함께 사용) |
| `--batch_size` | `4` | 배치 크기 (RTX 4060: 16 권장) |
| `--num_workers` | `4` | DataLoader 워커 수 (Windows 오류 시 `0` 으로 설정) |

> **VRAM 참고:** RTX 4060 (8GB) 기준 `--batch_size 16` 권장. OOM 발생 시 `8`로 낮출 것.

---

## 학습 결과 확인

```bash
# TensorBoard로 loss 모니터링
tensorboard --logdir ./output_sa
```

체크포인트는 `output_sa/checkpoint-{epoch}.pth` 로 20 epoch마다 저장.

---

## 베이스라인

이 프로젝트는 [DMAE (ICLR 2023)](https://github.com/quanlin-wu/dmae) 를 기반으로 합니다.

```bibtex
@inproceedings{wu2023dmae,
  title={Denoising Masked Autoencoders Help Robust Classification},
  author={Wu, QuanLin and Ye, Hang and Gu, Yuntian and Zhang, Huishuai and Wang, Liwei and He, Di},
  booktitle={ICLR},
  year={2023}
}
```
