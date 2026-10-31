"""Lightweight stand-in backend (OpenCV GrabCut) with the same interface as SAM2Backend.

Use it to try the review workflow on a laptop without PyTorch/SAM 2. It is a
colour-model segmenter, not a learned one: expect rougher masks and weaker tracking.
"""
import cv2
import numpy as np

WORK_SIDE = 512  # GrabCut runs on a downscaled copy for speed


class GrabCutBackend:
    name = "grabcut"

    def set_image(self, bgr):
        self.full_shape = bgr.shape[:2]
        self._small, self._scale = self._shrink(bgr)

    @staticmethod
    def _shrink(img):
        s = min(1.0, WORK_SIDE / max(img.shape[:2]))
        return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s

    def _up(self, mask_small):
        h, w = self.full_shape
        return cv2.resize(mask_small.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST) > 0

    def propose(self, points, labels, box):
        img, s = self._small, self._scale
        h, w = img.shape[:2]
        gc = np.full((h, w), cv2.GC_BGD, np.uint8)
        pos = [p for p, l in zip(points, labels) if l == 1]
        if box is not None:
            x, y, bw, bh = [int(v * s) for v in box]
            gc[max(0, y):y + bh, max(0, x):x + bw] = cv2.GC_PR_FGD
        elif pos:   # no box: search in a window around the positive clicks
            xs, ys = [p[0] * s for p in pos], [p[1] * s for p in pos]
            m = int(0.25 * max(h, w))
            gc[max(0, int(min(ys)) - m):int(max(ys)) + m,
               max(0, int(min(xs)) - m):int(max(xs)) + m] = cv2.GC_PR_FGD
        else:
            return [], []
        for (px, py), lab in zip(points, labels):
            cv2.circle(gc, (int(px * s), int(py * s)), 4,
                       cv2.GC_FGD if lab == 1 else cv2.GC_BGD, -1)
        bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
        try:
            cv2.grabCut(img, gc, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return [], []
        raw = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        # candidates: raw / largest blob / largest blob with holes filled
        n, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        cands = [raw > 0]
        if n > 1:
            big = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cands.append(big)
            cnts, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(raw)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            cands.append(filled > 0)
        return [self._up(c) for c in cands], [float("nan")] * len(cands)

    def propagate(self, frames_dir, first_mask):
        """Frame-to-frame GrabCut seeded by the previous mask (no learned tracking)."""
        from pathlib import Path
        files = sorted(Path(frames_dir).glob("*.jpg"))
        prev = cv2.resize(first_mask.astype(np.uint8), None, fx=self._scale,
                          fy=self._scale, interpolation=cv2.INTER_NEAREST)
        k = np.ones((9, 9), np.uint8)
        for pos, f in enumerate(files):
            if pos == 0:
                yield 0, first_mask
                continue
            img, _ = self._shrink(cv2.imread(str(f)))
            if prev.sum() < 20:
                yield pos, np.zeros(self.full_shape, bool)
                continue
            sure = cv2.erode(prev, k)
            near = cv2.dilate(prev, k)
            gc = np.full(prev.shape, cv2.GC_BGD, np.uint8)
            gc[near > 0] = cv2.GC_PR_BGD
            gc[prev > 0] = cv2.GC_PR_FGD
            gc[sure > 0] = cv2.GC_FGD
            try:
                cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)),
                            2, cv2.GC_INIT_WITH_MASK)
                cur = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
            except cv2.error:
                cur = prev
            prev = cur
            yield pos, self._up(cur)
