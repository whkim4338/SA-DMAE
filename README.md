# SA-DMAE: Spatial-Axial Denoising Masked Autoencoder

> 컴퓨터비전 수업 프로젝트 — 다반 7팀  
> DMAE를 2.5D 의료 영상 복원에 맞게 확장한 모델

---

## 프로젝트 구조

```
SA-DMAE/
├── models_sa_dmae.py       # SA-DMAE 모델 (핵심)
├── dataset_medical.py      # BraTS / EDG 의료 영상 데이터셋
├── main_pretrain_sa.py     # 학습 스크립트
├── colab_preprocess.ipynb  # Colab 전처리 노트북 (3D → 2.5D .pt 변환)
├── util/
│   ├── misc.py             # 학습 유틸 (torch 2.x 호환 수정)
│   ├── pos_embed.py        # Positional embedding (numpy 2.x 호환 수정)
│   └── lr_sched.py
└── requirements.txt
```

---

## 아키텍처 요약

```
Input (B, 3, 3, 224, 224)  ← 3장 연속 슬라이스 × 3채널(T1ce/T2/FLAIR)
        ↓
Normalize
        ↓
Spatial Stream (center slice + Gaussian noise + random masking)   ─┐
Axial Stream   (3 slices, no masking, shared ViT weights)          ─┘
        ↓
ViT Block × 8  (shared weights)
        ↓
[ViT Block → AxialAttentionBlock (Cross-Attn)] × 4
        ↓
Decoder (DMAE 동일)
        ↓
MSE Loss (center slice 기준)
```

- **Spatial Stream**: 기존 DMAE와 동일하게 중앙 슬라이스 복원
- **Axial Attention**: 인접 슬라이스의 같은 위치 토큰 간 상관관계 학습
- **Cross-Attention**: Spatial(Q) → Axial(KV) 로 문맥 융합

---

## 환경 설정

```bash
# 1. 가상환경 생성
python -m venv cv_env
cv_env\Scripts\activate        # Windows
# source cv_env/bin/activate   # Linux/Mac

# 2. 패키지 설치
pip install -r requirements.txt
```

---

## 데이터 준비

### 옵션 A: Colab 전처리 (권장)
1. BraTS 2021 데이터를 Google Drive에 업로드
2. `colab_preprocess.ipynb`를 Colab에서 실행
3. 생성된 `.pt` 파일을 로컬에 다운로드

### 옵션 B: 로컬 NIfTI 직접 사용
BraTS 2021 디렉토리 구조:
```
datasets/BraTS2021/
└── BraTS2021_00000/
    ├── BraTS2021_00000_t1ce.nii.gz
    ├── BraTS2021_00000_t2.nii.gz
    ├── BraTS2021_00000_flair.nii.gz
    └── BraTS2021_00000_seg.nii.gz
```

---

## 학습 실행

### .pt 파일로 학습 (Colab 전처리 후)
```bash
python main_pretrain_sa.py \
    --data_path ./pt_slices \
    --preprocessed \
    --device cpu \
    --model sa_dmae_vit_base_patch16 \
    --epochs 200 \
    --batch_size 4 \
    --accum_iter 4 \
    --sigma 0.25 \
    --mask_ratio 0.75
```

### NIfTI 직접 학습
```bash
python main_pretrain_sa.py \
    --data_path ./datasets/BraTS2021 \
    --device cpu \
    --model sa_dmae_vit_base_patch16 \
    --epochs 200 \
    --batch_size 4
```

### BraTS + EDG 합쳐서 학습
```bash
python main_pretrain_sa.py \
    --data_path ./datasets/BraTS2021 \
    --edg_path  ./datasets/EDG \
    --device cpu \
    --model sa_dmae_vit_base_patch16 \
    --epochs 200 \
    --batch_size 4
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `sa_dmae_vit_base_patch16` | 모델 크기 (base / large) |
| `--n_slices` | `3` | 연속 슬라이스 수 (홀수) |
| `--axial_depth` | `4` | Axial Attention 삽입 레이어 수 |
| `--sigma` | `0.25` | Gaussian noise 강도 |
| `--mask_ratio` | `0.75` | 마스킹 비율 |
| `--preprocessed` | `False` | .pt 파일 사용 여부 |

---

## 베이스라인

이 프로젝트는 [DMAE (ICLR 2023)](https://github.com/quanlin-wu/dmae) 를 기반으로 합니다.

```
@inproceedings{wu2023dmae,
  title={Denoising Masked Autoencoders Help Robust Classification},
  author={Wu, QuanLin and Ye, Hang and Gu, Yuntian and Zhang, Huishuai and Wang, Liwei and He, Di},
  booktitle={ICLR},
  year={2023}
}
```
