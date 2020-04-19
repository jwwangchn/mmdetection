from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class MaskRCNNFusion(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(MaskRCNNFusion, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        sar_img = img[:, 0:3, :, :]
        sar_feature = self.backbone(sar_img)

        if img.shape[1] == 6:
            rgb_img = img[:, 3:, :, :]
            rgb_feature = self.backbone(rgb_img)
        else:
            rgb_feature = sar_feature

        fusion_feature = []
        for sar_stage_feature, rgb_stage_feature in zip(sar_feature, rgb_feature):
            fusion_feature.append((sar_stage_feature + rgb_stage_feature) / 2.0)

        fusion_feature = tuple(fusion_feature)
        if self.with_neck:
            fusion_feature = self.neck(fusion_feature)
        return fusion_feature