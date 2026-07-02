# SGBiKD
<img src="Pics/Introduction.png" alt="Introduction" width="1200">

## DR(eye)VE-SOD Candidate Detector Comparison

This script provides a qualitative comparison of four object detectors used as
candidate generators for the DR(eye)VE-SOD annotation process:

- YOLO11x
- YOLO12x
- YOLO26x
- YOLOv8x-Worldv2

The purpose is to compare the quality of object proposals produced by each
detector under the same annotation conditions. It is not a comparison of human
annotators. Instead, each detector serves as a candidate-object generator, while
the same saliency-based attended-object criterion is applied to all models.

### Procedure

For sampled frames from DR(eye)VE sequences, the script:

1. Loads the Garmin RGB frame and its corresponding grayscale saliency map.
2. Runs each detector with a very low confidence threshold to obtain a broad
   pool of candidate object proposals.
3. Maps detections to the seven DR(eye)VE-SOD target classes:
   `people`, `car`, `motorcycle`, `traffic-light`, `traffic-sign`, `bus`, and
   `truck`.
4. Applies identical post-processing to every model, including class mapping,
   minimum-size and geometry filtering, and ego-vehicle hood suppression.
5. Retains a fixed number of high-confidence proposals per frame to ensure a
   comparable candidate pool across models.
6. Computes the saliency share of each detection as the fraction of total
   frame-level saliency contained inside its bounding box.
7. Marks an object as attended when its saliency share exceeds the selected
   threshold. When no candidate reaches the threshold, the highest-saliency
   candidates are selected as fallback attended objects.
8. Saves per-model visualizations and a 2×2 mosaic for direct comparison.

### Output Visualization

For every sampled frame, the generated overlays include:

- **Blue bounding boxes:** all retained candidate detections.
- **Green bounding boxes:** candidate detections selected as attended objects.
- **Saliency heatmap overlay:** the corresponding driver-attention map.
- **2×2 model mosaic:** side-by-side comparison of the four candidate detectors.

The output is saved under:

```text
compare_vis_optionA_4models/
├── per_model/
│   ├── yolo11x/
│   ├── yolo12x/
│   ├── yolo26x/
│   └── yolov8x-worldv2/
└── mosaic/

# Ablation Code Layout

This folder collects the ablation-facing code into a cleaner structure without
breaking the original scripts in the repo root.

## Structure

- `teachers/`
  - wrappers for the saliency teachers used in KD experiments
  - currently includes `W3DA` and `S-ViT / SCF-ViT`
- `students/`
  - student-detector builders and validation helpers
- `kd/`
  - reusable KD head definitions
  - reverse-KD helper functions and scheduling utilities
- `experiments/`
  - clean entrypoints that forward to the existing experiment scripts

## Recommended entrypoints

- `python -m ablation.experiments.compare_teachers`
- `python -m ablation.experiments.compare_students`
- `python -m ablation.experiments.profile_inference_only`
- `python -m ablation.experiments.profile_deploy_only`
- `python -m ablation.experiments.profile_kd_heads`
- `python -m ablation.experiments.profile_students_only`
