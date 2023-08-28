# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

import numpy as np
from detectron2.layers import nms
from detectron2.structures import Instances, Boxes

def plot_proposals_on_image(image_list, proposals_rpn):
    from PIL import Image, ImageDraw
    import cv2 as cv
    from torchvision.transforms.functional import to_pil_image
    for index in range(len(image_list)):

        # Extract the image tensor from the ImageList
        image_tensor = image_list[index].contiguous()

        # Convert the image tensor to a PIL image for visualization
        pil_image = to_pil_image(image_tensor.cpu())

         # Create a drawing context
        draw = ImageDraw.Draw(pil_image)

        # Extract the proposal_boxes
        rpn_boxes = proposals_rpn[index].proposal_boxes.tensor
        rpn_scores = proposals_rpn[index].objectness_logits
        nms_threshold = 0.5

        # NMS rpn_boxes
        nms_index = nms(rpn_boxes, rpn_scores, nms_threshold)
        nms_boxes = proposals_rpn[index][nms_index]

        # Loop through the proposals
        count = 0
        for box, k  in zip(nms_boxes.proposal_boxes.tensor, nms_boxes.k):
        # for box  in nms_boxes.proposal_boxes.tensor:
            if count == 10: break
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1,y1+20), f"{k}", fill=(0,0,255))
            count += 1
        
        # Convert PIL Image to OpenCV format
        cv_image = np.array(pil_image)

        # Show the image using OpenCV
        cv.imshow('Image with Proposals', cv_image)

        # Wait for user to press any key
        cv.waitKey(0)

        # save image
        path = "/home/mmlab206/Pictures/"
        cv.imwrite(f"{path}rpn_{index}.png", cv_image)
        
        # Close the image window
        cv.destroyAllWindows()

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNNv2(GeneralizedRCNN):
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
        """
            # keys: p2, p3, p4, p5, p6
            # p2:torch.Size([4, 256, 296, 200])
            # p3:torch.Size([4, 256, 148, 100])
            # p4:torch.Size([4, 256, 74, 50])
            # p5:torch.Size([4, 256, 37, 25])
            # p6:torch.Size([4, 256, 19, 13])
        """

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

            return losses, [], [], None, features

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

            return {}, proposals_rpn, proposals_roih, ROI_predictions,features

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

        elif branch == "CL":

            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )
            # plot_proposals_on_image(images, proposals_rpn)
            for index in range(len(proposals_rpn)):
                # Extract the proposal_boxes
                rpn_boxes = proposals_rpn[index].proposal_boxes.tensor
                rpn_scores = proposals_rpn[index].objectness_logits
                nms_threshold = 0.5

                # NMS rpn_boxes
                nms_index = nms(rpn_boxes, rpn_scores, nms_threshold)

                filtered_proposals = Instances(proposals_rpn[index].image_size)
                filtered_proposals.proposal_boxes = Boxes(rpn_boxes[nms_index])
                filtered_proposals.objectness_logits = rpn_scores[nms_index]
                proposals_rpn[index] = filtered_proposals

            return proposals_rpn, features


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
