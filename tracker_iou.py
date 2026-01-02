import time

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, (area_a + area_b - inter))

class IoUTracker:
    """
    Assigns stable IDs by matching detections to previous tracks using IoU.
    Good enough to replicate your 'movement threshold' + per-person timers.
    """
    def __init__(self, iou_match=0.35, max_age_sec=1.0):
        self.iou_match = float(iou_match)
        self.max_age_sec = float(max_age_sec)
        self.next_id = 1
        self.tracks = {}  # id -> {"box":..., "t":..., "class_id":..., "conf":...}

    def update(self, detections):
        now = time.time()

        # drop stale tracks
        stale = [tid for tid, tr in self.tracks.items() if (now - tr["t"]) > self.max_age_sec]
        for tid in stale:
            self.tracks.pop(tid, None)

        assigned = set()
        out = []

        # greedy matching
        for det in detections:
            best_tid, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                v = iou(det["box"], tr["box"])
                if v > best_iou:
                    best_iou, best_tid = v, tid

            if best_tid is not None and best_iou >= self.iou_match:
                tid = best_tid
            else:
                tid = self.next_id
                self.next_id += 1

            self.tracks[tid] = {
                "box": det["box"],
                "class_id": det["class_id"],
                "conf": det["conf"],
                "t": now,
            }
            assigned.add(tid)

            out.append({
                "id": tid,
                "box": det["box"],
                "class_id": det["class_id"],
                "conf": det["conf"],
            })

        return out

