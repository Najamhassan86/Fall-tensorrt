import time
from collections import deque, defaultdict

class EventLogic:
    def __init__(
        self,
        fps=30,
        fall_class_id=0,
        sit_class_id=1,
        movement_threshold=20.0,
        fall_window=10,
        fall_required=6,
        enable_postfall=True,
        postfall_window_sec=5,
        lying_aspect_thresh=0.55,   # h/w threshold
        lying_frames_required=10,
        immobile_sec_required=3,
        sit_limit_min=10,
        alert_cooldown_sec=20,
    ):
        self.fps = int(fps)
        self.fall_class_id = int(fall_class_id)
        self.sit_class_id = int(sit_class_id)
        self.movement_threshold = float(movement_threshold)

        self.fall_buf = deque(maxlen=int(fall_window))
        self.fall_required = int(fall_required)

        self.enable_postfall = bool(enable_postfall)
        self.postfall_window_sec = float(postfall_window_sec)
        self.lying_aspect_thresh = float(lying_aspect_thresh)
        self.lying_frames_required = int(lying_frames_required)
        self.immobile_sec_required = float(immobile_sec_required)

        self.sit_limit_sec = float(sit_limit_min) * 60.0
        self.alert_cooldown_sec = float(alert_cooldown_sec)

        self.last_alert_ts = 0.0
        self.postfall_state = defaultdict(lambda: {"t0": None, "lying": 0, "immobile_t0": None})
        self.sit_start = {}  # tid -> time.time()

        # movement state for each ID
        self.pos_hist = defaultdict(lambda: deque(maxlen=10))

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _avg_movement(self, tid):
        pts = self.pos_hist[tid]
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            total += (dx*dx + dy*dy) ** 0.5
        return total / max(1, len(pts))

    def _postfall_confirm(self, tid, box, movement, now, is_falling):
        st = self.postfall_state[tid]

        if is_falling and st["t0"] is None:
            st["t0"] = now
            st["lying"] = 0
            st["immobile_t0"] = None

        if st["t0"] is None:
            return False

        if (now - st["t0"]) > self.postfall_window_sec:
            st["t0"] = None
            st["lying"] = 0
            st["immobile_t0"] = None
            return False

        # lying evidence
        x1, y1, x2, y2 = box
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        aspect = h / w
        if aspect < self.lying_aspect_thresh:
            st["lying"] += 1

        # immobility evidence
        if movement <= self.movement_threshold:
            if st["immobile_t0"] is None:
                st["immobile_t0"] = now
        else:
            st["immobile_t0"] = None

        lying_ok = st["lying"] >= self.lying_frames_required
        immobile_ok = (st["immobile_t0"] is not None) and ((now - st["immobile_t0"]) >= self.immobile_sec_required)
        return lying_ok or immobile_ok

    def update(self, tracked_dets):
        """
        tracked_dets: list of dicts with {id, box, class_id, conf}
        returns: dict events {fall_confirmed:bool, sit_too_long:bool, should_alert:bool}
        """
        now = time.time()

        raw_fall = any(int(d["class_id"]) == self.fall_class_id for d in tracked_dets) if tracked_dets else False
        self.fall_buf.append(bool(raw_fall))
        confirmed_transition = (sum(self.fall_buf) >= self.fall_required)

        confirmed_postfall_any = False
        sit_too_long = False

        for d in tracked_dets or []:
            tid = int(d["id"])
            box = d["box"]
            cid = int(d["class_id"])

            # movement tracking
            self.pos_hist[tid].append(self._center(box))
            movement = self._avg_movement(tid)

            # sitting timer (per person)
            if cid == self.sit_class_id:
                if tid not in self.sit_start:
                    self.sit_start[tid] = now
                if (now - self.sit_start[tid]) >= self.sit_limit_sec:
                    sit_too_long = True
            else:
                self.sit_start.pop(tid, None)

            # post-fall confirm per track
            if self.enable_postfall:
                is_falling = (cid == self.fall_class_id)
                if self._postfall_confirm(tid, box, movement, now, is_falling):
                    confirmed_postfall_any = True

        fall_confirmed = confirmed_transition and (confirmed_postfall_any if self.enable_postfall else True)

        can_alert = (now - self.last_alert_ts) >= self.alert_cooldown_sec
        should_alert = fall_confirmed and can_alert
        if should_alert:
            self.last_alert_ts = now

        return {
            "fall_confirmed": bool(fall_confirmed),
            "sit_too_long": bool(sit_too_long),
            "should_alert": bool(should_alert),
        }

