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

## Citing Unbiased Teacher

If you use Unbiased Teacher in your research or wish to refer to the results published in the paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{liu2021unbiased,
    title={Unbiased Teacher for Semi-Supervised Object Detection},
    author={Liu, Yen-Cheng and Ma, Chih-Yao and He, Zijian and Kuo, Chia-Wen and Chen, Kan and Zhang, Peizhao and Wu, Bichen and Kira, Zsolt and Vajda, Peter},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021},
}
```

Also, if you use Detectron2 in your research, please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

## License

This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.

# Acknowledgements
We use Unbiased-teacher official code as our baseline. 
- [Unbiased-Teacher](https://github.com/facebookresearch/unbiased-teacher)
