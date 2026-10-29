# TRI2I Thermal-to-RGB Image Translation

**Project:** Foundation Model–Based Visible–Thermal Translation and Scene Understanding  

**Module:** Thermal Infrared (TIR) to Visible RGB Image Translation  
**Dataset:** TRI2I  
**Framework:** PyTorch  
**Translation Model:** Pix2Pix Conditional GAN  
**Development Status:** Baseline implemented; further evaluation and improvement in progress

---

## 1. Overview

Thermal infrared (TIR) cameras capture heat-related information and can operate in low-light conditions. However, thermal images generally do not contain the same visual appearance, colours, and textures as conventional visible-light RGB images.

This module investigates paired thermal-to-RGB image translation using the TRI2I dataset. A Pix2Pix conditional generative adversarial network (cGAN) is trained to generate an RGB-like image conditioned on a thermal input.

The module is part of a larger project that aims to combine cross-spectral image translation with scene understanding, including object and person detection.

**Important:** The generated image is a model prediction. It does not guarantee recovery of the scene's true colours or details that were not captured by the thermal sensor.

## 2. Objectives

- Prepare and verify paired thermal and visible RGB images.
- Establish a reproducible training, validation, and testing pipeline.
- Train a Pix2Pix model for thermal-to-RGB translation.
- Compare saved checkpoints using quantitative metrics.
- Inspect generated images against thermal inputs and ground-truth RGB images.
- Establish a documented baseline for subsequent model improvements.
- Provide a foundation for integration with a scene-understanding pipeline.

## 3. Dataset

The experiment uses the TRI2I paired image dataset.

In the original dataset archive, the verified folder convention is:

| Original folder | Modality | Role |
|---|---|---|
| `class1` | Visible RGB | Target image |
| `class2` | Thermal infrared (TIR) | Input image |

The images are 256 × 256 pixels. The original dataset contains 148,829 verified matching image pairs.

### Dataset split

A deterministic split was created using random seed 42.

| Subset | Paired images | Percentage |
|---|---:|---:|
| Training | 119,063 | 80% |
| Validation | 14,882 | 10% |
| Test | 14,884 | 10% |
| **Total** | **148,829** | **100%** |

The split manifest is stored separately from the source code. The test subset is reserved for final evaluation and should not be used to select checkpoints or tune hyperparameters.

### Data preparation

The preparation workflow included:

1. Inspecting the original dataset archive.
2. Verifying matching RGB and thermal filenames.
3. Checking image readability and dimensions.
4. Creating a reproducible split using seed 42.
5. Preparing the paired input and target images for Pix2Pix.
6. Verifying the data-loader output shapes.

The original dataset archive and split manifest are stored separately from this repository. Dataset access and redistribution must comply with the applicable licence and terms.

## 4. Methodology

### 4.1 Paired image translation

For each training pair, the thermal image is the source input and the corresponding visible RGB image is the target.

The model learns a mapping:

`Thermal image → Pix2Pix generator → Generated RGB image`

The generator attempts to reproduce visible appearance conditioned on the thermal input. The discriminator distinguishes real paired RGB targets from generated outputs during adversarial training.

### 4.2 Model architecture

The baseline uses the Pix2Pix conditional GAN architecture.

| Component | Configuration |
|---|---|
| Framework | PyTorch |
| Translation method | Pix2Pix conditional GAN |
| Generator | U-Net 256 |
| Discriminator | Basic PatchGAN configuration |
| Input channels | 3 |
| Output channels | 3 |
| Image resolution | 256 × 256 |
| Normalization | Batch normalization |
| GAN objective | Vanilla GAN loss |
| Reconstruction objective | L1 loss |
| L1 loss weight | 100 |

The three-channel input represents the thermal image in the prepared RGB-channel format. This does not mean that the source thermal sensor captures three visible colour channels.

### 4.3 Training configuration

| Parameter | Value |
|---|---|
| Batch size | 4 |
| Optimizer | Adam |
| Initial learning rate | 0.0002 |
| Beta1 | 0.5 |
| L1 loss coefficient | 100 |
| Load size | 256 × 256 |
| Crop size | 256 × 256 |
| Learning-rate schedule | Initial constant phase followed by decay |
| Checkpointing | Epoch checkpoints and latest checkpoints |

The recorded training configuration used a constant-learning-rate phase of 10 epochs followed by a 10-epoch decay phase. However, the available logs do not establish that the full schedule completed successfully. The saved checkpoints are therefore treated as experimental baselines, not as evidence of a completed 20-epoch training run.

## 5. Evaluation metrics

The experiment uses the following image-quality metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute pixel difference. Lower values are better.
- **Peak Signal-to-Noise Ratio (PSNR):** Measures pixel-level reconstruction quality on a logarithmic scale. Higher values are better.
- **Structural Similarity Index (SSIM):** Measures structural similarity between generated and reference images. Higher values are better.
- **LPIPS (optional):** A perceptual similarity metric that can supplement pixel-based metrics when available.

These metrics measure different aspects of image quality. No single metric fully captures semantic correctness, realistic colours, or usefulness for downstream detection.

## 6. Validation results

Four saved generator checkpoints were evaluated using the same 14,882 validation pairs.

| Checkpoint | Mean MAE ↓ | Mean PSNR ↑ | Mean SSIM ↑ |
|---|---:|---:|---:|
| Baseline Epoch 1 | 0.101973 | 16.0545 dB | 0.477500 |
| Baseline Epoch 2 | 0.088573 | 16.8708 dB | 0.508279 |
| Baseline Latest | 0.086238 | 17.0482 dB | 0.506095 |
| Resume Latest | 0.085606 | 16.9864 dB | 0.498972 |

### Interpretation

- Baseline Latest achieved the highest mean PSNR among the four evaluated checkpoints.
- Resume Latest achieved the lowest mean MAE.
- Baseline Epoch 2 achieved the highest mean SSIM.
- The metrics do not all favour the same checkpoint.

Baseline Latest was selected as the **provisional candidate** for final held-out test evaluation. This selection is based on its validation PSNR and competitive MAE; it does not establish that it is universally the best checkpoint.

## 7. Previously recorded test results

A separate evaluation of the Resume Latest checkpoint was completed on all 14,884 test pairs.

| Metric | Resume Latest test result |
|---|---:|
| Mean MAE | 0.085790 |
| Mean PSNR | 16.9817 dB |
| Mean SSIM | 0.498237 |
| Test images evaluated | 14,884 |

**These results belong to Resume Latest. They are not the final test results for Baseline Latest.** The held-out test evaluation of the selected Baseline Latest checkpoint remains pending.

## 8. Qualitative observations

Visual inspection of representative thermal inputs, generated images, and ground-truth RGB targets showed that the baseline can preserve some scene structure and prominent bright regions.

The current outputs also exhibit limitations:

- Dark or underexposed generated regions.
- Blurred textures and fine details.
- Differences in colour and illumination compared with the reference.
- Imperfect reconstruction of road markings and background structures.
- Potential visual artifacts.

These observations indicate that the baseline is useful for establishing a research starting point but requires further investigation before claiming high-fidelity translation.

## 9. Reproducibility and artifacts

The experiment artifacts are stored separately from this repository, including the original dataset archive, split manifest, checkpoints, and evaluation files.

Relevant artifacts include:

- `tri2i_split_seed42.json`
- `TRI2I_checkpoint_validation_comparison.csv`
- `validation_baseline_epoch1_metrics.csv`
- `validation_baseline_epoch2_metrics.csv`
- `validation_baseline_latest_metrics.csv`
- `validation_resume_latest_metrics.csv`
- `THERMAL_RGB_BASELINE_REPORT.txt`

The original dataset and large model checkpoints are not included in this source folder.

When adding files to this repository, keep the following distinctions clear:

- Training and evaluation code should be version-controlled.
- Evaluation metrics and reports should document the corresponding checkpoint.
- Visual examples should identify their input, generated prediction, and reference image.
- Model weights should only be redistributed when permitted.
- Dataset files should only be redistributed when permitted by the dataset terms.

## 10. Limitations

The current baseline has several limitations:

1. Generated RGB appearance may differ from the true visible scene.
2. Pixel-based metrics may not fully represent perceptual quality.
3. Some scenes exhibit low contrast, blur, or reconstruction artifacts.
4. Performance may vary across scene types and illumination conditions.
5. The effect of image translation on downstream object detection has not yet been established.
6. The final test evaluation for the provisional Baseline Latest checkpoint remains pending.

## 11. Planned work

The next development stages are:

1. Complete held-out test evaluation of Baseline Latest.
2. Save and document its final test metrics.
3. Inspect representative test predictions, including failure cases.
4. Investigate controlled model improvements using validation results.
5. Develop the reverse RGB-to-Thermal translation branch.
6. Audit suitable labelled datasets for object and person detection.
7. Develop and evaluate the scene-understanding component.
8. Integrate translation and detection into a single demonstration.
9. Prepare the final experimental report and project presentation.

## 12. Repository organisation

The intended module structure is:

```text
Base_Model/
└── Thermal_RGB_TRI2I/
    ├── README.md
    ├── notebooks/
    ├── results/
    ├── samples/
    └── inference/
```

Only create the subfolders when the corresponding files are available. Keep large datasets, temporary files, and restricted model artifacts out of the public source repository.

## 13. Conclusion

This experiment establishes a paired Pix2Pix baseline for thermal-to-RGB image translation using the TRI2I dataset. Dataset pairing and split preparation were verified, multiple generator checkpoints were evaluated on a common validation set, and representative generated images were inspected.

The results provide a reproducible starting point for further cross-spectral translation research. Final test evaluation of the selected Baseline Latest checkpoint, additional model improvements, reverse-direction translation, and integration with scene understanding remain part of the planned development.

---

**Module status:** Baseline implemented; final evaluation and further development in progress.
