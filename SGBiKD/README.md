# SGBiKD Paper Code Map

This folder reorganizes the code for the paper:

`Driver-Aware Salient Object Detection via Bidirectional Knowledge Distillation`

It is meant to collect the paper-specific workflow into one place:

- dataset annotation
- saliency-teacher training/evaluation
- SGBiKD runs
- ablation-facing entrypoints

## Dataset annotation

- DR(eye)VE-SOD:
  - [annotation/annotate_dreyeve_sod.py](annotation/annotate_dreyeve_sod.py)
- TrafficGaze-SOD:
  - [annotation/annotate_trafficgaze_sod.py](annotation/annotate_trafficgaze_sod.py)

Both scripts export `Y` / `A` label files compatible with the downstream KD
evaluation scripts in this repo.

Notes:

- The DR(eye)VE annotator follows the saliency-share, temporal consistency, and
  temporal propagation rules from the paper and exports labels into `labels_Y12`
  by default.
- The TrafficGaze annotator uses fixation maps when they exist under
  `fixationframe/`. If those maps are missing, it derives sparse pseudo-fixation
  points from the saliency maps so the pipeline remains runnable.

## Teacher model code

- Train / test SCF-ViT on DR(eye)VE:
  - [teacher/train_scf_vit_dreyeve.py](teacher/train_scf_vit_dreyeve.py)
- Evaluate SCF-ViT on TrafficGaze:
  - [teacher/eval_scf_vit_trafficgaze.py](teacher/eval_scf_vit_trafficgaze.py)

## SGBiKD / SOD code

- DR(eye)VE teacher comparison:
  - [sod/run_dreyeve_teacher_comparison.py](sod/run_dreyeve_teacher_comparison.py)
- DR(eye)VE student comparison:
  - [sod/run_dreyeve_student_comparison.py](sod/run_dreyeve_student_comparison.py)
- TrafficGaze SGBiKD with SCF-ViT:
  - [sod/run_trafficgaze_sgbikd_svit.py](sod/run_trafficgaze_sgbikd_svit.py)
- TrafficGaze W3DA transfer / BiKD:
  - [sod/run_trafficgaze_sgbikd_w3da.py](sod/run_trafficgaze_sgbikd_w3da.py)

## Analysis / visualization

- Confusion matrix and class analysis:
  - [analysis/run_confusion_matrix.py](analysis/run_confusion_matrix.py)
- Qualitative visualization:
  - [analysis/run_visualization.py](analysis/run_visualization.py)

## Notes

- These entrypoints wrap the working scripts already present in the repo, so you
  can use the paper-oriented layout without breaking older commands.
- Some comparison tables in the PDF mix exact paper settings with exploratory
  scripts developed later. This folder points to the closest implemented code
  paths in the current workspace.
- Recommended usage is with module execution, for example:
  - `python -m paper_sgbikd.annotation.annotate_dreyeve_sod`
  - `python -m paper_sgbikd.annotation.annotate_trafficgaze_sod`
  - `python -m paper_sgbikd.sod.run_trafficgaze_sgbikd_svit`
