#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups. 
    """
    cfg = get_cfg() # return a detectron2 CfgNode instance
    add_ubteacher_config(cfg) # 加入unbiased teacher 架構
    cfg.merge_from_file(args.config_file)
    # 從 configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml 加入 1% COCO 資料集的設定
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # 設定logger, log 環境資訊、cmdline參數、config、備份config到輸出資料夾
    return cfg


def main(args):
    cfg = setup(args)

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            # 
            # res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    # print(f"\nCommand Line Args: {args}\n")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),	
    )

"""
python train_net.py \
    --num-gpus 2 \
    --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
    SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16
"""
"""
python train_net.py \
    --num-gpus 2 \
    --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
    SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 \
    SEMISUPNET.BURN_UP_STEP 200000 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    SOLVER.BASE_LR 0.001 \
    MODEL.WEIGHTS byol_default_bs64_224_oneTenth_woINpretrain.pth
"""
"""
# evaluate
python train_net.py \
    --eval-only \
    --num-gpus 2 \
    --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
    SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 \
    MODEL.WEIGHTS output/model_0039999.pth
"""
"""
# evaluate
python train_net.py \
    --eval-only \
    --num-gpus 2 \
    --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
    SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 \
    SEMISUPNET.Trainer baseline \
    MODEL.WEIGHTS byol_results/byol_resnet50_FPN_bs16_0ep.pth 
"""