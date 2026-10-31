"""SAM 2 wrapper: propose masks on one frame, then propagate through the video.

Needs PyTorch + the `sam2` package and a downloaded checkpoint (see README).
"""
from pathlib import Path
import numpy as np

# model size -> (config name inside the sam2 package, default checkpoint file)
MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


class SAM2Backend:
    name = "sam2"

    def __init__(self, model_size="large", checkpoint=None, device=None):
        try:
            import torch
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise SystemExit(
                "SAM 2 is not installed. Install PyTorch, then\n"
                "  pip install \"git+https://github.com/facebookresearch/sam2.git\"\n"
                "or run with --backend grabcut to try the workflow without SAM 2."
            ) from exc
        self._torch = torch
        self._build_video = build_sam2_video_predictor
        self.config, default_ckpt = MODELS[model_size]
        self.checkpoint = Path(checkpoint or Path("checkpoints") / default_ckpt)
        if not self.checkpoint.is_file():
            raise SystemExit(
                f"Checkpoint not found: {self.checkpoint}\n"
                "Download one, e.g.\n"
                f"  https://dl.fbaipublicfiles.com/segment_anything_2/092824/{default_ckpt}\n"
                "and put it in ./checkpoints/ (or pass --checkpoint)."
            )
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        print(f"Loading SAM 2 ({model_size}) on {self.device} ...")
        self._image_predictor = SAM2ImagePredictor(
            build_sam2(self.config, str(self.checkpoint), device=self.device))
        self._video_predictor = None

    def _ctx(self):
        """Inference mode (+ bfloat16 autocast on CUDA, as in the SAM 2 demos)."""
        import contextlib
        stack = contextlib.ExitStack()
        stack.enter_context(self._torch.inference_mode())
        if self.device.type == "cuda":
            stack.enter_context(self._torch.autocast("cuda", dtype=self._torch.bfloat16))
        return stack

    # -- step 3: propose a mask on the first frame -------------------------
    def set_image(self, bgr):
        rgb = bgr[:, :, ::-1].copy()
        with self._ctx():
            self._image_predictor.set_image(rgb)

    def propose(self, points, labels, box):
        """points: [(x, y)], labels: [1|0], box: (x, y, w, h) or None.

        Returns (list of boolean masks best-first, list of scores).
        """
        pc = np.array(points, dtype=np.float32).reshape(-1, 2) if points else None
        pl = np.array(labels, dtype=np.int32) if points else None
        b = None
        if box is not None:
            x, y, w, h = box
            b = np.array([x, y, x + w, y + h], dtype=np.float32)
        # SAM 2 recommends a single mask once the prompt is no longer ambiguous.
        multimask = (len(points) + (box is not None)) <= 1
        with self._ctx():
            masks, scores, _ = self._image_predictor.predict(
                point_coords=pc, point_labels=pl, box=b, multimask_output=multimask)
        order = np.argsort(-scores)
        return [masks[i] > 0 for i in order], [float(scores[i]) for i in order]

    # -- step 8: apply the approved mask to the whole video -----------------
    def propagate(self, frames_dir, first_mask):
        """Yield (position, boolean mask) for every frame in frames_dir (in order)."""
        if self._video_predictor is None:
            self._video_predictor = self._build_video(
                self.config, str(self.checkpoint), device=self.device)
        pred = self._video_predictor
        with self._ctx():
            state = pred.init_state(video_path=str(frames_dir),
                                    offload_video_to_cpu=True)
            pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=first_mask)
            for pos, _obj_ids, logits in pred.propagate_in_video(state):
                yield pos, (logits[0, 0] > 0.0).cpu().numpy()
            pred.reset_state(state)
