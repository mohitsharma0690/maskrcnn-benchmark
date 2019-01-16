BATCH_SIZE=64
args=(
  # --expert_path ./h5_trajs/mujoco_trajs/normal_hopper/rebuttal_hopper_100/
  --config-file "configs/ms_cutting/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"

  # OUTPUT_DIR "./results/max_iter_1000_period_1000_lr_.0025_im_per_batch_4/"

  SOLVER.IMS_PER_BATCH 4
  SOLVER.BASE_LR 0.0025
  SOLVER.MAX_ITER 10000
  SOLVER.STEPS "(480000, 640000)"
  SOLVER.CHECKPOINT_PERIOD 1000
  TEST.IMS_PER_BATCH 1

  MODEL.MS_CHECKPOINT_WEIGHTS_TO_REMOVE '["roi_heads.box.predictor.cls_score.weight", "roi_heads.box.predictor.cls_score.bias", "roi_heads.box.predictor.bbox_pred.weight", "roi_heads.box.predictor.bbox_pred.bias"]'
  MODEL.ROI_BOX_HEAD.NUM_CLASSES 3

)

echo "${args[@]}"

python -m pdb tools/ms_train_net.py "${args[@]}"
