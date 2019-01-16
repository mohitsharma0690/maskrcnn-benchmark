BATCH_SIZE=64

args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"

  SOLVER.IMS_PER_BATCH 2
  SOLVER.BASE_LR 0.0025
  SOLVER.MAX_ITER 720000
  SOLVER.STEPS "(480000, 640000)"
  SOLVER.CHECKPOINT_PERIOD 5
  TEST.IMS_PER_BATCH 1

  MODEL.MS_CHECKPOINT_WEIGHTS_TO_REMOVE '["roi_heads.box.predictor.cls_score.weight", "roi_heads.box.predictor.cls_score.bias", "roi_heads.box.predictor.bbox_pred.weight", "roi_heads.box.predictor.bbox_pred.bias"]'

)

echo "${args[@]}"

python -m pdb tools/ms_train_net.py "${args[@]}"
