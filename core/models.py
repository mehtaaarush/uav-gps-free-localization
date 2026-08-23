"""
core/models.py
==============
The two transformer models that power the system.

ROLE 1 — Global retrieval  : DINOv2 ViT-S/14   (view-invariant embeddings)
ROLE 2 — Local matching     : EfficientLoFTR     (detector-free dense matcher)

Both load lazily and cache. If a model or its weights cannot be downloaded
(offline machine), the code falls back to a lighter alternative and prints a
clear warning, so the pipeline still runs end-to-end for the demo.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════════════════════
# ROLE 1 — DINOv2 retrieval backbone
# ════════════════════════════════════════════════════════════════════════════

class RetrievalBackbone(nn.Module):
    """
    Produces one L2-normalised embedding vector per image.

    Primary  : DINOv2 ViT-S/14  (384-dim)  — self-supervised, view-invariant.
               Best for cross-view satellite↔UAV retrieval.
    Fallback : EfficientNet-B0 via timm (1280-dim) if DINOv2 can't be loaded.
    """

    def __init__(self, prefer: str = "dinov2"):
        super().__init__()
        self.kind = None
        self.dim  = None
        self.model = None

        if prefer == "dinov2":
            try:
                # DINOv2 from torch.hub (facebookresearch/dinov2)
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", verbose=False
                ).to(DEVICE).eval()
                self.kind = "dinov2"
                self.dim  = 384
                print("[RetrievalBackbone] Loaded DINOv2 ViT-S/14 (dim=384)")
            except Exception as e:
                warnings.warn(f"DINOv2 unavailable ({e}); "
                              f"falling back to EfficientNet-B0")

        if self.model is None:
            self._load_efficientnet()

        for p in self.parameters():
            p.requires_grad = False

    def _load_efficientnet(self):
        try:
            import timm
            m = timm.create_model("efficientnet_b0", pretrained=True)
            self.model = nn.Sequential(*list(m.children())[:-1],
                                       nn.AdaptiveAvgPool2d(1)).to(DEVICE).eval()
            self.kind = "efficientnet"
            self.dim  = 1280
            print("[RetrievalBackbone] Loaded EfficientNet-B0 (dim=1280)")
        except Exception:
            import torchvision.models as tvm
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            self.model = nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
            self.kind = "resnet18"
            self.dim  = 512
            print("[RetrievalBackbone] Loaded ResNet-18 fallback (dim=512)")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 3, H, W) normalised tensor.
            For DINOv2, H and W must be multiples of 14.
        returns : (N, dim) float32, L2-normalised
        """
        x = x.to(DEVICE)
        if self.kind == "dinov2":
            feat = self.model(x)                 # (N, 384)
        else:
            feat = self.model(x).flatten(1)      # (N, dim)
        feat = nn.functional.normalize(feat, dim=1)
        return feat.cpu()


# ════════════════════════════════════════════════════════════════════════════
# ROLE 2 — EfficientLoFTR dense matcher
# ════════════════════════════════════════════════════════════════════════════

class DenseMatcher:
    """
    Finds dense pixel correspondences between two images.

    Primary  : EfficientLoFTR (HuggingFace transformers) — detector-free,
               works on textureless regions where ORB/SIFT fail.
    Fallback : OpenCV ORB + BFMatcher if transformers/weights unavailable.

    .match(img0_bgr, img1_bgr) -> (pts0, pts1, scores)
        pts0 : (M, 2) matched points in img0 (x, y)
        pts1 : (M, 2) matched points in img1 (x, y)
    """

    def __init__(self, prefer: str = "efficientloftr"):
        self.kind = None
        self.processor = None
        self.model = None

        if prefer == "efficientloftr":
            try:
                # NOTE: must use AutoModelForKeypointMatching, NOT AutoModel.
                # AutoModel loads the feature-extractor backbone (BackboneOutput,
                # no .matches); the KeypointMatching head returns matches/keypoints.
                from transformers import (AutoImageProcessor,
                                          AutoModelForKeypointMatching)
                name = "zju-community/efficientloftr"
                self.processor = AutoImageProcessor.from_pretrained(name)
                self.model = (AutoModelForKeypointMatching
                              .from_pretrained(name).to(DEVICE).eval())
                self.kind = "efficientloftr"
                print("[DenseMatcher] Loaded EfficientLoFTR (detector-free)")
            except Exception as e:
                warnings.warn(f"EfficientLoFTR unavailable ({e}); "
                              f"falling back to ORB")

        if self.model is None:
            import cv2
            self._cv2 = cv2
            self.orb = cv2.ORB_create(nfeatures=2000)
            self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.kind = "orb"
            print("[DenseMatcher] Loaded ORB fallback")

    @torch.no_grad()
    def match(self, img0_bgr: np.ndarray, img1_bgr: np.ndarray):
        if self.kind == "efficientloftr":
            return self._match_eloftr(img0_bgr, img1_bgr)
        return self._match_orb(img0_bgr, img1_bgr)

    def _match_eloftr(self, img0_bgr, img1_bgr):
        import cv2
        from PIL import Image
        im0 = Image.fromarray(cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB))
        im1 = Image.fromarray(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB))

        # processor expects a list of an image-pair: [[im0, im1]]
        inputs = self.processor(images=[[im0, im1]],
                                return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)

        # target_sizes: (batch, 2, 2) = original (H, W) of each image in the pair
        sizes = torch.tensor([[[im0.height, im0.width],
                               [im1.height, im1.width]]])
        proc = self.processor.post_process_keypoint_matching(
            outputs, sizes, threshold=0.2
        )[0]

        pts0   = proc["keypoints0"].cpu().numpy()
        pts1   = proc["keypoints1"].cpu().numpy()
        scores = proc["matching_scores"].cpu().numpy()
        return pts0, pts1, scores

    def _match_orb(self, img0_bgr, img1_bgr):
        cv2 = self._cv2
        k0, d0 = self.orb.detectAndCompute(img0_bgr, None)
        k1, d1 = self.orb.detectAndCompute(img1_bgr, None)
        if d0 is None or d1 is None or len(d0) < 4 or len(d1) < 4:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))
        matches = sorted(self.bf.match(d0, d1), key=lambda m: m.distance)[:200]
        pts0 = np.float32([k0[m.queryIdx].pt for m in matches])
        pts1 = np.float32([k1[m.trainIdx].pt for m in matches])
        scores = np.float32([1.0 - m.distance / 256.0 for m in matches])
        return pts0, pts1, scores
