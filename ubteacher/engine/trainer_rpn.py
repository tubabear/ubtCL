# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler
from detectron2.layers import nms
from torch.nn import functional as F

def plot_proposals_on_image(image_list, proposals_rpn, cl, number):
    from PIL import Image, ImageDraw
    import cv2 as cv
    from torchvision.transforms.functional import to_pil_image
    cl_count = 0
    category_file = "./datasets/coco/coco_category_.txt"
    with open(category_file, 'r') as f:
        lines = f.readlines()
        lines = [line.split('\t') for line in lines]
        catPairs = [line[1] for line in lines]
    for index in range(len(image_list)):

        # Extract the image tensor from the ImageList
        image_tensor = image_list[index]["image"].contiguous()

        # Convert the image tensor to a PIL image for visualization
        pil_image = to_pil_image(image_tensor.cpu())

         # Create a drawing context
        draw = ImageDraw.Draw(pil_image)

        # Extract the proposal_boxes
        rpn_boxes = proposals_rpn[index].proposal_boxes.tensor
        # rpn_boxes = proposals_rpn[index].gt_boxes.tensor
        # rpn_classes = proposals_rpn[index].gt_classes
        rpn_scores = proposals_rpn[index].objectness_logits
        # rpn_scores = proposals_rpn[index].scores

        # Loop through the proposals
        count = 0
        # for box, clas, score in zip(rpn_boxes, rpn_classes, rpn_scores):
        for box, score in zip(rpn_boxes, rpn_scores):
            if count == number: break
            count+=1
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=2)
            # draw.text((x1,y1+20), f"{catPairs[int(clas)]}-{score:.2f}", fill=(0,0,255))
            draw.text((x1,y1+20), f"{score:.2f}", fill=(0,0,255))
            # cl_score = cl[cl_count].mean()
            # cl_count += 1
            # draw.text((x1,y1+5), f"{cl_score:.2f}", fill=(0,255,0))
        
        # Convert PIL Image to OpenCV format
        # cv_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)
        cv_image = np.array(pil_image)

        # Show the image using OpenCV
        cv.imshow('Image with Proposals', cv_image)

        # Wait for user to press any key
        cv.waitKey(0)

        # save image
        path = "/home/mmlab206/Pictures/"
        cv.imwrite(f"{path}RPN{number}_{index}_.png", cv_image)
        
        # Close the image window
        cv.destroyAllWindows()

def report_gpu_usage():
    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / 1e9
    cached = torch.cuda.memory_reserved(current_device) / 1e9
    if current_device == 0:
        print(f'{allocated:.2f} GB allocated, {cached:.2f} GB reserved (cached)')

from torch import nn
def Proj_MLP2D(dim, projection_size, hidden_size=2048):
    # siamese : 3 layers, BN every layers, output fc no ReLU, same dimension
    return nn.Sequential(
        nn.Conv2d(dim, hidden_size, kernel_size=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_size, projection_size, kernel_size=1),
        nn.BatchNorm2d(projection_size),
    )

def Pred_MLP2D(dim, projection_size, hidden_size=512):
    # siamese : 2 layers, BN at hidden layer, output fc no ReLU and BN, one fourth dimension
    return nn.Sequential(
        nn.Conv2d(dim, hidden_size, kernel_size=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_size, projection_size, kernel_size=1),
    )

def BN_ReLU2d(dim):
    return nn.Sequential(
        nn.BatchNorm2d(dim),
        # nn.ReLU(inplace=True),
        # nn.Identity(),
        )

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    
    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
    
    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter) # list[dict] -> #batch
        # dict_keys(['file_name', 'height', 'width', 'image_id', 'image', 'instances'])
        # for d in data:
        #     print(f"{d['height']} x {d['width']} -> {d['image'].shape}")

        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
            # return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


# Unbiased Teacher Trainer
class UBTeacherTrainerV2(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # DefaultTrainer: 產生一個簡單的預設訓練器(使用config定義的model, optimizer, dataloader, LR schedfuler)
        # auto_scale_workers: 根據cfg.SOLVER.REFERENCE_WORLD_SIZE決定有幾個workeres，如果跟輸入的number of workers數量不同，則自動調整成每個GPU有相同的batch size (IMS_PER_BATCH//REGERENCE_WORLD_SIZE)
        
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        # <class 'ubteacher.modeling.meta_arch.rcnn.TwoStagePseudoLabGeneralizedRCNN'>

        optimizer = self.build_optimizer(cfg, model)

        if cfg.CONTRASTIVE.PREDICTOR:
            print(f"Create predictors")
            self.predictor = Pred_MLP2D(256,256,hidden_size=64).cuda()
            optimizer.add_param_group({"params":self.predictor.parameters()})
        
        if cfg.CONTRASTIVE.PROJECTOR:
            self.projector = Proj_MLP2D(256,256,hidden_size=256).cuda()
        else: # only BatchNorm and ReLU layer
            self.projector = BN_ReLU2d(256).cuda()

        optimizer.add_param_group({"params":self.projector.parameters()})
        
        # resume pp
        if "pp_checkpoint.pt" in os.listdir(cfg.OUTPUT_DIR):
            print("load pp checkpoint")
            ck_path = os.path.join(cfg.OUTPUT_DIR, "pp_checkpoint.pt")
            pp_ck = torch.load(ck_path)
            self.projector.load_state_dict(pp_ck["projector"])
            if cfg.CONTRASTIVE.PREDICTOR:
                self.predictor.load_state_dict(pp_ck["predictor"])

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
                    # find_unused_parameters=True
            )
            if cfg.CONTRASTIVE.PREDICTOR:
                self.predictor = DistributedDataParallel(self.predictor, device_ids=[comm.get_local_rank()], broadcast_buffers=False )
            if cfg.CONTRASTIVE.PROJECTOR:
                self.projector = DistributedDataParallel(self.projector, device_ids=[comm.get_local_rank()], broadcast_buffers=False )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            # if TORCH_VERSION >= (1, 7):
            #     self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            # return PascalVOCDetectionEvaluator(dataset_name)
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well    
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    ROI_predictions_k,
                    features_t,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )

            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            #  add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            # unlabel_data_k = self.add_label(
            #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            # )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            record_all_unlabel_data, _, _, _, features_s= self.model(
                all_unlabel_data, branch="supervised"
            )


            filtered_proposals_rpn = self.object_threhsold(
                proposals_rpn_unsup_k,
                num_threshold=0.5
                )

            # contrastive learning with unlabel data
            contrastive_loss = self.featureCL(
                target_feature=features_t,
                online_feature=features_s,
                boxes=filtered_proposals_rpn,
                # boxes=nms_rois,
                img_k=unlabel_data_k,
                # img_q=unlabel_data_q,
                )
            del features_t
            torch.cuda.empty_cache()

            record_all_label_data, _, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
            record_dict.update(new_record_all_unlabel_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values()) + contrastive_loss
            record_dict["contrastive_loss"] = contrastive_loss.item()

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        if comm.get_local_rank() == 0 :
            if ((self.iter+1) % self.cfg.SOLVER.CHECKPOINT_PERIOD) == 0:
                self.save_pp_checkpoint()

        # for k,v in self.model_teacher.state_dict().items():
        #     if not torch.all(torch.eq(v, self.model.state_dict()[f"module.{k}"])):
        #         print(f"{self.iter} after backward -> ", k)

        # dict_keys(['state', 'param_groups'])
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

    def contrastive_learning(self, target_view, online_view, target_net, online_net, mode="train"):

        # create avgpool layer
        if self.cfg.CONTRASTIVE.AVGPOOL:
            avgpool = torch.nn.AvgPool2d(self.cfg.CONTRASTIVE.AVGPOOL)
        else:
            avgpool = torch.nn.AvgPool2d(1)

        branch = "CL"
        in_features = self.cfg.MODEL.ROI_HEADS.IN_FEATURES

        # create box pooler
        try:
            box_pooler_t = target_net.module.roi_heads.box_pooler
        except AttributeError:
            box_pooler_t = target_net.roi_heads.box_pooler

        rpn_num = self.cfg.CONTRASTIVE.RPN
        rpn_threshold = self.cfg.CONTRASTIVE.RPN_THRESHOLD
        # traget branch
        with torch.no_grad():
            # Extract feature
            proposals_t, features_t = target_net(target_view, branch=branch) # list, dict
            features_t = [features_t[f] for f in in_features]

            if rpn_num: # only use top rpn_num nms proposals 
                proposals_boxes = [x.proposal_boxes[:rpn_num] for x in proposals_t]
            elif rpn_threshold:
                proposals_boxes = []
                for proposal in proposals_t:
                    valid_map = proposal.objectness_logits > rpn_threshold
                    proposals_boxes.append(proposal.proposal_boxes[valid_map])
            else: # use all proposals
                proposals_boxes = [x.proposal_boxes for x in proposals_t]

            # RoI align
            roi_features_t = box_pooler_t(features_t, proposals_boxes) 
            # tensor:[400, 256, 7, 7]

            # Dense or Pooling
            if not self.cfg.CONTRASTIVE.DENSE: # avgpool2d
                roi_features_t = avgpool(roi_features_t.detach())

            roi_features_t = self.projector(roi_features_t.detach())

        # create boox pooler
        try:
            box_pooler_s = online_net.module.roi_heads.box_pooler
        except AttributeError:
            box_pooler_s = online_net.roi_heads.box_pooler
        
        # online branch 
        if mode == "warm":
            # freeze backbone
            with torch.no_grad():
                _, features_s = online_net(online_view, branch=branch)
                features_s = [features_s[f] for f in in_features]
                roi_features_s = box_pooler_s(features_s, proposals_boxes) 
        else:
            _, features_s = online_net(online_view, branch=branch)
            features_s = [features_s[f] for f in in_features]
            roi_features_s = box_pooler_s(features_s, proposals_boxes)

        if not self.cfg.CONTRASTIVE.DENSE: # avgpool2d
            roi_features_s = avgpool(roi_features_s)


        # freeze projector, predictor if "train"
        if mode =="train": 
            with torch.no_grad():
                roi_features_s = self.projector(roi_features_s)
                if self.cfg.CONTRASTIVE.PREDICTOR:
                    roi_features_s = self.predictor(roi_features_s)
        else:
            roi_features_s = self.projector(roi_features_s)
            if self.cfg.CONTRASTIVE.PREDICTOR:
                roi_features_s = self.predictor(roi_features_s)

        p1 = roi_features_s
        z2 = roi_features_t
        cos_sim = torch.nn.functional.cosine_similarity(p1, z2.detach())
        contrastive_loss = (2-2*cos_sim).mean()

        # record proposals information
        if (comm.get_local_rank() == 0) and ((self.iter+1) % 100 == 0) :
            coverage_area = 0
            average_area = 0
            average_score = 0
            count = 0
            for p in proposals_t:
                w, h = p.image_size
                if rpn_num:
                    boxes = p.proposal_boxes.tensor[:rpn_num,:]
                    scores = p.objectness_logits[:rpn_num]
                else:
                    boxes = p.proposal_boxes.tensor
                    scores = p.objectness_logits
                # 計算每個boxes的寬度和高度
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]

                # 計算每個boxes的面積
                areas = widths * heights

                # 計算總面積和平均面積
                total_area = areas.sum()
                average_area += total_area / len(boxes)
                coverage_area += total_area / ( w * h )
                average_score += scores.sum() / len(boxes)

            record_file = os.path.join(self.cfg.OUTPUT_DIR,"CL_info.csv")
            with open(record_file,'a') as f:
                f.write(f"{self.iter+1},")
                f.write(f"{coverage_area/len(proposals_t)},")
                f.write(f"{average_area/len(proposals_t)},")
                f.write(f"{average_score/len(proposals_t)},")
                f.write("\n")
                # record area%, average confidence

        return contrastive_loss

    def featureCL(self, target_feature, online_feature, boxes, img_k):

        # create avgpool layer
        if self.cfg.CONTRASTIVE.AVGPOOL:
            avgpool = torch.nn.AvgPool2d(self.cfg.CONTRASTIVE.AVGPOOL)
        else:
            avgpool = torch.nn.AvgPool2d(1)

        in_features = self.cfg.MODEL.ROI_HEADS.IN_FEATURES

        # create box pooler
        try:
            box_pooler = self.model.module.roi_heads.box_pooler
        except AttributeError:
            box_pooler = self.model.roi_heads.box_pooler

        rpn_num = self.cfg.CONTRASTIVE.RPN
        rpn_threshold = self.cfg.CONTRASTIVE.RPN_THRESHOLD
        # traget branch
        with torch.no_grad():
            features_t = [target_feature[f] for f in in_features]

            if rpn_num: # only use top rpn_num nms proposals 
                proposals_boxes = [x.proposal_boxes[:rpn_num] for x in boxes]
            elif rpn_threshold:
                proposals_boxes = []
                for proposal in boxes:
                    valid_map = proposal.objectness_logits > rpn_threshold
                    proposals_boxes.append(proposal.proposal_boxes[valid_map])
            else: # use all proposals
                proposals_boxes = [x.proposal_boxes for x in boxes]
                if comm.get_local_rank() == 0 and self.iter % 100 == 0:
                    box_numbers = [len(x) for x in proposals_boxes]
                    self.record_info(f"{sum(box_numbers)/len(box_numbers)}, proposals")

            # RoI align
            roi_features_t = box_pooler(features_t, proposals_boxes) 
            # tensor:[400, 256, 7, 7]

            # Dense or Pooling
            if not self.cfg.CONTRASTIVE.DENSE: # avgpool2d
                roi_features_t = avgpool(roi_features_t.detach())

            roi_features_t = self.projector(roi_features_t.detach())

        # online branch 
        features_s = [online_feature[f] for f in in_features]
        roi_features_s = box_pooler(features_s, proposals_boxes)

        if not self.cfg.CONTRASTIVE.DENSE: # avgpool2d
            roi_features_s = avgpool(roi_features_s)

        roi_features_s = self.projector(roi_features_s)
        if self.cfg.CONTRASTIVE.PREDICTOR:
            roi_features_s = self.predictor(roi_features_s)

        p1 = roi_features_s
        z2 = roi_features_t
        cos_sim = torch.nn.functional.cosine_similarity(p1, z2.detach())
        contrastive_loss = (2-2*cos_sim).mean()
        # plot_proposals_on_image(img_k, boxes, contrastive_loss, 10)
        # plot_proposals_on_image(img_k, boxes, contrastive_loss, 100)

        return contrastive_loss

    def record_info(self, infos):
        record_file = os.path.join(self.cfg.OUTPUT_DIR,"CL_info.csv")
        with open(record_file,'a') as f:
            f.write(f"{self.iter+1},")
            f.write(f"{infos}")
            f.write("\n")
            # record area%, average confidence

    def object_threhsold(self, proposals_rpn, num_threshold=0.5, threshold=None):
        filtered_proposals_rpn = []
        for index in range(len(proposals_rpn)):
            # filter by confidence
            if threshold:
                conf_index = proposals_rpn[index].objectness_logits > threshold
                proposals_rpn[index] = proposals_rpn[index][conf_index]

            # NMS by object logitness
            nms_index = nms(
                proposals_rpn[index].proposal_boxes.tensor,
                proposals_rpn[index].objectness_logits,
                num_threshold)

            filtered_proposals_rpn.append(proposals_rpn[index][nms_index])
            
        return filtered_proposals_rpn

    def save_pp_checkpoint(self):
        pp_state_dict = {}
        if isinstance(self.projector, (DistributedDataParallel, DataParallel)):
                pp_state_dict["projector"] = self.projector.module.state_dict()
        else:
            pp_state_dict["projector"] = self.projector.state_dict()
        
        if self.cfg.CONTRASTIVE.PREDICTOR:
            if isinstance(self.predictor, (DistributedDataParallel, DataParallel)):
                pp_state_dict["predictor"] = self.predictor.module.state_dict()
            else:
                pp_state_dict["predictor"] = self.predictor.state_dict()
        ck_path = os.path.join(self.cfg.OUTPUT_DIR, "pp_checkpoint.pt")
        torch.save(pp_state_dict, ck_path)

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                if not torch.all(torch.eq(student_model_dict[key],value)):
                    new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                    )
                else:
                    new_teacher_dict[key] = value

            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)
    
    @torch.no_grad()
    def _update_student_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            teacher_model_dict = {
                f"module.{key}": value for key, value in self.model_teacher.state_dict().items()
            }
        else:
            teacher_model_dict = self.model_teacher.state_dict()

        new_student_dict = OrderedDict()
        for key, value in self.model.state_dict().items():
            if key in teacher_model_dict.keys():
                if not torch.all(torch.eq(teacher_model_dict[key], value)):
                    new_student_dict[key] = (
                        teacher_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                    )
                else:
                    new_student_dict[key] = value

            else:
                raise Exception("{} is not found in teacher model".format(key))

        self.model.load_state_dict(new_student_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
