"""KLT feature tracking front-end (OpenCV) producing MSCKF-ready pixel tracks.

Pipeline per frame: pyramidal Lucas-Kanade forward tracking with a
forward-backward consistency check, RANSAC fundamental-matrix outlier
rejection against the previous frame, and Shi-Tomasi re-detection (masked away
from live tracks) when the track count drops. Output pixels are undistorted to
the ideal pinhole model so they are consistent with qlio.camera.CameraModel.
"""

import cv2
import numpy as np


class KLTTracker:
    def __init__(self, camera, dist_coeffs, max_corners=250, quality=0.01,
                 min_distance=25, replenish_below=180, fb_thresh=1.0,
                 ransac_px=1.0):
        self.K = np.array([[camera.fx, 0, camera.cx],
                           [0, camera.fy, camera.cy],
                           [0, 0, 1.0]])
        self.dist = np.asarray(dist_coeffs, dtype=np.float64)
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.replenish_below = replenish_below
        self.fb_thresh = fb_thresh
        self.ransac_px = ransac_px
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.prev = None
        self.pts = np.empty((0, 2), np.float32)   # distorted pixel positions
        self.ids = np.empty((0,), np.int64)
        self._next_id = 0

    def _detect(self, img, mask_pts):
        mask = np.full(img.shape, 255, np.uint8)
        for x, y in mask_pts:
            cv2.circle(mask, (int(x), int(y)), self.min_distance, 0, -1)
        n = self.max_corners - len(mask_pts)
        if n <= 0:
            return np.empty((0, 2), np.float32)
        c = cv2.goodFeaturesToTrack(img, n, self.quality, self.min_distance,
                                    mask=mask, blockSize=7)
        return np.empty((0, 2), np.float32) if c is None else c.reshape(-1, 2)

    def step(self, img):
        """Track into `img`; returns {feature_id: undistorted pixel (2,)}."""
        if self.prev is not None and len(self.pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, img, self.pts, None, **self.lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.prev, nxt, None, **self.lk)
            fb = np.linalg.norm(back - self.pts, axis=1)
            ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < self.fb_thresh)
            h, w = img.shape
            ok &= (nxt[:, 0] >= 0) & (nxt[:, 0] < w) & (nxt[:, 1] >= 0) & (nxt[:, 1] < h)
            prev_ok, nxt_ok, ids_ok = self.pts[ok], nxt[ok], self.ids[ok]
            if len(nxt_ok) >= 8:
                _F, inl = cv2.findFundamentalMat(prev_ok, nxt_ok, cv2.FM_RANSAC,
                                                 self.ransac_px, 0.999)
                if inl is not None:
                    keep = inl.reshape(-1) == 1
                    nxt_ok, ids_ok = nxt_ok[keep], ids_ok[keep]
            self.pts, self.ids = nxt_ok.astype(np.float32), ids_ok
        if len(self.pts) < self.replenish_below:
            fresh = self._detect(img, self.pts)
            if len(fresh):
                new_ids = np.arange(self._next_id, self._next_id + len(fresh))
                self._next_id += len(fresh)
                self.pts = np.vstack([self.pts, fresh]).astype(np.float32)
                self.ids = np.concatenate([self.ids, new_ids])
        self.prev = img
        if not len(self.pts):
            return {}
        und = cv2.undistortPoints(self.pts.reshape(-1, 1, 2), self.K, self.dist,
                                  P=self.K).reshape(-1, 2)
        return {int(fid): und[k].astype(np.float64) for k, fid in enumerate(self.ids)}
