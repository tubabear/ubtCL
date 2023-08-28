# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.transforms.functional import to_pil_image

def plot_proposals_on_image(image_list, proposals_rpn):
    for index in range(len(image_list)):
        fig, ax = plt.subplots(1)

        # Extract the image tensor from the ImageList
        image_tensor = image_list[index].contiguous()

        # Convert the image tensor to a PIL image for visualization
        pil_image = to_pil_image(image_tensor.cpu())

        # Display the image
        ax.imshow(np.array(pil_image))

        # Plot each proposal on the image
        count = 0
        for proposal, obj_log in zip(proposals_rpn[index].proposal_boxes, proposals_rpn[index].objectness_logits):
            if count >= 10: break
            count += 1
        # for proposal, score in zip(proposals_rpn[index].pred_boxes, proposals_rpn[index].scores):
            # print(score)
            x, y, x2, y2 = proposal.cpu().numpy()
            rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()
        plt.waitforbuttonpress()
        plt.close()

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # keys: p2, p3, p4, p5, p6
        # p2:torch.Size([4, 256, 296, 200])
        # p3:torch.Size([4, 256, 148, 100])
        # p4:torch.Size([4, 256, 74, 50])
        # p5:torch.Size([4, 256, 37, 25])
        # p6:torch.Size([4, 256, 19, 13])

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # list, dict
            # len(proposals_rpn):batch_size
            # len(proposals_rpn[i]):1000
            """
            Instances(
                num_instances=1000, image_height=672, image_width=1007, 
                fields=[proposal_boxes: Boxes(tensor(
                [[1.6575,  497.0330,  139.2980,  548.7043],
                ...,
                [ 387.2868,  478.6669,  407.5875,  499.1975]])), 
            objectness_logits: tensor([8.9663, 8.5458,..., 1.4468])])
            """
            # proposal_losses.keys:{loss_rpn_cls, loss_rpn_loc}


            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )
            # list, dict
            # len(_):batch_size
            """
            Instances(
            num_instances=512, image_height=704, image_width=939, 
            fields=[proposal_boxes: 
                Boxes(tensor([[0.0000, 19.2395, 46.4457, 80.6991], ... 
                objectness_logits: tensor([6.7278e+00, ...
                gt_classes: tensor([ 9,...,
                gt_boxes: Boxes(tensor([[0, 22.7157, 38.1797, 82.7153],...
            )

            """
            # len(_[0]):512
            # detector_losses.keys:loss_cls, loss_box_reg

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            # list , tuple
            # proposals_roih[0]:
            """
            Instances(num_instances=52, image_height=640, image_width=853, fields=[pred_boxes: Boxes(tensor([
            [232.4892, 325.2491, 282.4567, 390.8666],...
            scores: tensor([0.9139...
            pred_classes: tensor([17, 17,...

            """
            # len(ROI_predictions):2
            # ROI_predictions[0].shape:4000,81 # scores(N,k+1)
            # ROI_predictions[1].shape:4000,81 # box_delta (N,k*4)
            # plot_proposals_on_image(images, proposals_rpn)


            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
