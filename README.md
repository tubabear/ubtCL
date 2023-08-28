# Enhancing Semi-Supervised Object Detection with Contrastive Learning

<p align="center">
</p>

# Installtion & Setup
We follow the installation precess of Unbiased Teacher official repo (https://github.com/facebookresearch/unbiased-teacher)

## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Training

- Train the Unbiased Teacher under 10% COCO-supervision

```shell
python train_net.py \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 \
       MODEL.META_ARCHITECTURE "TwoStagePseudoLabGeneralizedRCNNv2" \
       CONTRASTIVE.DENSE True \
       CONTRASTIVE.PREDICTOR True \
       CONTRASTIVE.PROJECTOR True \
       CONTRASTIVE.RPN 10 \
```
## License

This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.

# Acknowledgements
We use Unbiased-teacher official code as our baseline. 
- [Unbiased-Teacher](https://github.com/facebookresearch/unbiased-teacher)
