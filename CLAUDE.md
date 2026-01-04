# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GVHMR (Gravity-View Human Motion Recovery) is a research project for world-grounded human motion recovery from videos. The codebase uses PyTorch Lightning for training, Hydra for configuration management, and integrates multiple components including visual odometry, 2D pose estimation, feature extraction, and SMPL/SMPLX body models.

## Development Setup

### Environment Installation
```bash
conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

### Required Checkpoints
The system requires several pretrained checkpoints in `inputs/checkpoints/`:
- `body_models/smplx/SMPLX_{GENDER}.npz` - SMPLX models (requires signup at SMPL-X website)
- `body_models/smpl/SMPL_{GENDER}.pkl` - SMPL models (requires signup at SMPL website)
- `gvhmr/gvhmr_siga24_release.ckpt` - Main GVHMR checkpoint
- `hmr2/epoch=10-step=25000.ckpt` - HMR2 pretrained weights
- `vitpose/vitpose-h-multi-coco.pth` - VitPose for 2D keypoints
- `yolo/yolov8x.pt` - YOLO for person detection

Downloadable checkpoints are available from Google Drive (see docs/INSTALL.md).

## Common Commands

### Demo/Inference
```bash
# Run demo on a video (static camera)
python tools/demo/demo.py --video=<path_to_video> -s

# Run demo with camera motion estimation (SimpleVO by default)
python tools/demo/demo.py --video=<path_to_video>

# Run demo with DPVO for camera motion (slower)
python tools/demo/demo.py --video=<path_to_video> --use_dpvo

# Run demo with custom focal length (for zoomed cameras)
python tools/demo/demo.py --video=<path_to_video> --f_mm=77

# Process entire folder
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### Training
```bash
# Train GVHMR model (default: 2x4090 GPUs, 420 epochs)
python tools/train.py exp=gvhmr/mixed/mixed

# Training adjusts batch size and GPU count in exp config files
# See hmr4d/configs/exp/gvhmr/mixed/mixed.yaml for details
```

### Testing
```bash
# Test on all benchmarks (3DPW, RICH, EMDB)
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# Test on individual datasets
python tools/train.py global/task=gvhmr/test_3dpw exp=gvhmr/mixed/mixed ckpt_path=<checkpoint>
python tools/train.py global/task=gvhmr/test_rich exp=gvhmr/mixed/mixed ckpt_path=<checkpoint>
python tools/train.py global/task=gvhmr/test_emdb exp=gvhmr/mixed/mixed ckpt_path=<checkpoint>
```

## Architecture Overview

### Core Pipeline
The GVHMR pipeline consists of several sequential stages:

1. **Preprocessing** (tools/demo/demo.py:run_preprocess)
   - Person detection and tracking (YOLO-based Tracker)
   - 2D pose extraction (VitPose)
   - Visual feature extraction (ViT from HMR2)
   - Camera motion estimation (SimpleVO or DPVO)

2. **GVHMR Model** (hmr4d/model/gvhmr/)
   - Input: image features, 2D keypoints, camera angular velocity, bounding boxes
   - Network: Relative transformer (hmr4d/network/gvhmr/relative_transformer.py)
   - Output: SMPLX parameters in both camera and world coordinates

3. **Post-processing** (hmr4d/model/gvhmr/utils/postprocess.py)
   - Static joint smoothing
   - Inverse kinematics refinement
   - Foot contact handling

### Key Components

**hmr4d/model/gvhmr/gvhmr_pl.py**: Main PyTorch Lightning training module
- Handles training loop with data augmentation
- Implements multi-dataset training (AMASS, BEDLAM, H36M, 3DPW)

**hmr4d/model/gvhmr/gvhmr_pl_demo.py**: Inference-only Lightning module
- Simplified interface for demo/inference
- Includes flip-test for better results

**hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py**: Core prediction pipeline
- Encoder/decoder for normalizing inputs/outputs
- Denoiser3D network for pose estimation
- Camera parameter handling

**hmr4d/utils/preproc/**: Preprocessing utilities
- `Tracker`: YOLO-based person tracking
- `VitPoseExtractor`: 2D pose estimation
- `Extractor`: ViT feature extraction
- `SimpleVO`: Simple visual odometry (default)
- `relpose/`: Alternative DPVO-based camera estimation (optional)

**hmr4d/network/**: Neural network architectures
- `gvhmr/relative_transformer.py`: Main transformer architecture
- `hmr2/`: HMR2.0 components (ViT backbone, SMPL head)
- `base_arch/`: Shared components (transformer layers, embeddings)

### Configuration System

Uses Hydra with a hierarchical config structure:
- `hmr4d/configs/train.yaml`: Training entry point
- `hmr4d/configs/demo.yaml`: Demo entry point
- `hmr4d/configs/exp/`: Experiment-specific configs
- `hmr4d/configs/global/task/`: Task definitions (train/test)
- Config composition: defaults → exp → global/task overrides

### Dataset Structure

Training expects preprocessed data in `inputs/` directory:
- `AMASS/hmr4d_support/`: Motion capture data
- `BEDLAM/hmr4d_support/`: Synthetic training data
- `H36M/hmr4d_support/`: Human3.6M dataset
- `3DPW/hmr4d_support/`: 3DPW test set
- `EMDB/hmr4d_support/`: EMDB test set
- `RICH/hmr4d_support/`: RICH test set

Each dataset has custom loaders in `hmr4d/dataset/`.

### Coordinate Systems

The codebase uses multiple coordinate frames:
- **Camera space** (`_c` suffix): Standard camera coordinates
- **Centered camera** (`_cr` suffix): Camera coords with root joint at origin
- **World space** (`_w` or `_ay` suffix): Absolute world coordinates with gravity-aligned Y-axis
- **Gravity-view** (`_ayfz` suffix): World coords with specific frontal facing

Key transformations in `hmr4d/utils/geo_transform.py` and `hmr4d/utils/geo/`.

### Body Models

Uses SMPL/SMPLX body models:
- `hmr4d/utils/body_model/`: Body model implementations
- `make_smplx()`: Factory function for different SMPL variants
- Outputs vertices and joints from pose/shape parameters

### Rendering

`hmr4d/utils/vis/renderer.py`: PyTorch3D-based mesh rendering
- In-camera rendering: Overlays on original video
- Global rendering: Third-person view with ground plane

## Code Style

- **Line length**: 120 characters (configured in pyproject.toml)
- **Type hints**: Not strictly enforced but used in some modules
- **Logging**: Use `hmr4d.utils.pylogger.Log` for consistent logging
- **Tensor naming**: Follow `<data>_<space>` convention (e.g., `verts_c`, `j3d_ay`)

## Important Notes

- **SimpleVO vs DPVO**: SimpleVO is now the default for camera estimation (faster, no extra dependencies). DPVO is optional and requires additional installation steps.
- **GPU requirements**: Training configuration assumes 2x4090 GPUs. Adjust `pl_trainer.devices` and batch size for different setups.
- **Flip-test**: Demo uses flip-test by default (processes video normally and flipped, then averages) for better results.
- **Video I/O**: Uses `imageio` with `pyav` backend for efficient video processing.
