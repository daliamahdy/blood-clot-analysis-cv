"""Low-key review window (plain OpenCV, no extra dependencies).

Left panel: the first frame with your prompts. Right panel: the proposed mask.
The logic lives in ReviewSession (testable without a display); run_review()
is only the thin window/event loop.
"""
import cv2
import numpy as np
from .visualization import overlay_mask

WIN = "Clot review"
HELP = ("L-click: +point | R-click: -point | b + drag: box | n/p: next/prev candidate | "
        "u: undo | c: clear | a/ENTER: APPROVE | q/ESC: quit")


class ReviewSession:
    def __init__(self, frame, backend, max_width=1700, points=None, labels=None, box=None):
        self.frame, self.backend = frame, backend
        h, w = frame.shape[:2]
        self.scale = min(1.0, max_width / (2 * w))
        self.pw, self.ph = int(w * self.scale), int(h * self.scale)
        self.points, self.labels = list(points or []), list(labels or [])
        self.box = box
        self.candidates, self.scores, self.idx = [], [], 0
        self.box_mode, self._drag, self._drag_now = False, None, None
        self.message = "Click on the clot to get a first proposal (or press b and drag a box)."
        backend.set_image(frame)
        self.refresh()

    # -- state -------------------------------------------------------------
    @property
    def mask(self):
        return self.candidates[self.idx] if self.candidates else None

    def refresh(self):
        if not self.points and self.box is None:
            self.candidates, self.scores, self.idx = [], [], 0
            return
        self.candidates, self.scores = self.backend.propose(self.points, self.labels, self.box)
        self.idx = 0
        if not self.candidates:
            self.message = "No mask for these prompts - add/undo a point or redraw the box."
        else:
            self.message = f"{len(self.candidates)} candidate(s). Approve with 'a' or refine the prompts."

    def prompts(self):
        return {"points": [list(map(float, p)) for p in self.points],
                "labels": [int(l) for l in self.labels],
                "box_xywh": list(map(float, self.box)) if self.box else None,
                "candidate_index": self.idx,
                "score": (self.scores[self.idx] if self.scores else None)}

    # -- input ---------------------------------------------------------------
    def _to_frame(self, x, y):
        return (x % self.pw) / self.scale, y / self.scale   # click on either panel

    def on_mouse(self, event, x, y, flags=0):
        fx, fy = self._to_frame(x, y)
        if self.box_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drag = self._drag_now = (fx, fy)
            elif event == cv2.EVENT_MOUSEMOVE and self._drag:
                self._drag_now = (fx, fy)
            elif event == cv2.EVENT_LBUTTONUP and self._drag:
                (x0, y0), (x1, y1) = self._drag, (fx, fy)
                self._drag = self._drag_now = None
                self.box_mode = False
                if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
                    self.box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
                    self.refresh()
                else:
                    self.message = "Box too small - press b and drag again."
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.points.append((fx, fy))
            self.labels.append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            self.refresh()

    def on_key(self, key):
        """Returns 'approve', 'quit' or None."""
        k = chr(key).lower() if 0 <= key < 128 else ""
        if k == "a" or key in (13, 10):
            if self.mask is None or not self.mask.any():
                self.message = "Nothing to approve yet - add a prompt first."
                return None
            return "approve"
        if k == "q" or key == 27:
            return "quit"
        if k == "n" and self.candidates:
            self.idx = (self.idx + 1) % len(self.candidates)
        elif k == "p" and self.candidates:
            self.idx = (self.idx - 1) % len(self.candidates)
        elif k == "u":
            if self.points:
                self.points.pop(); self.labels.pop()
            elif self.box is not None:
                self.box = None
            self.refresh()
        elif k == "c":
            self.points, self.labels, self.box = [], [], None
            self.refresh()
            self.message = "Cleared."
        elif k == "b":
            self.box_mode = True
            self.message = "Box mode: drag a rectangle around the clot."
        return None

    # -- drawing ---------------------------------------------------------
    def render(self):
        s, size = self.scale, (self.pw, self.ph)
        left = cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA)
        for (px, py), lab in zip(self.points, self.labels):
            col = (0, 220, 0) if lab else (0, 0, 255)
            cv2.circle(left, (int(px * s), int(py * s)), 7, col, -1)
            cv2.circle(left, (int(px * s), int(py * s)), 8, (255, 255, 255), 1)
        box = self.box
        if self._drag and self._drag_now:
            (x0, y0), (x1, y1) = self._drag, self._drag_now
            box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        if box:
            x, y, w, h = [int(v * s) for v in box]
            cv2.rectangle(left, (x, y), (x + w, y + h), (0, 255, 255), 2)

        right = cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA)
        if self.mask is not None:
            m = cv2.resize(self.mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
            right = overlay_mask(right, m)
        title = "Proposed mask"
        if self.candidates:
            sc = self.scores[self.idx]
            title += f"  candidate {self.idx + 1}/{len(self.candidates)}"
            title += f"  (score {sc:.2f})" if sc == sc else ""
            title += f"  area {int(self.mask.sum())} px"
        self._label(left, "Original (first frame)")
        self._label(right, title)

        bar = np.full((52, 2 * self.pw, 3), 32, np.uint8)
        cv2.putText(bar, self.message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.putText(bar, HELP, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 180, 180), 1)
        return np.vstack([np.hstack([left, right]), bar])

    @staticmethod
    def _label(img, text):
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)


def run_review(frame, backend, points=None, labels=None, box=None):
    """Open the window. Returns (mask, prompts) if approved, else (None, None)."""
    session = ReviewSession(frame, backend, points=points, labels=labels, box=box)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, lambda e, x, y, f, _: session.on_mouse(e, x, y, f))
    result = None
    try:
        while result is None:
            cv2.imshow(WIN, session.render())
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                result = "quit"
            elif key != 255:
                result = session.on_key(key)
    finally:
        cv2.destroyWindow(WIN)
    if result == "approve":
        return session.mask.copy(), session.prompts()
    return None, None
