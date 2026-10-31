import numpy as np

class SimpleSortTracker:
    """
    Lightweight IoU-based multi-object tracker for vehicle tracking and speed estimation.
    Self-contained without requiring external C++/Fortran dependencies.
    """
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        # tracks dictionary: id -> track state
        self.tracks = {}

    @staticmethod
    def _iou(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        areaB = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        """
        detections: list or np.ndarray of [x1, y1, x2, y2, score, class_id]
        returns: list of [x1, y1, x2, y2, track_id, class_id]
        """
        updated_tracks = []
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return np.empty((0, 6))

        matched = set()
        det_matched = set()

        # Match existing tracks with current detections based on maximum IoU
        for tid, track in self.tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for d_idx, det in enumerate(detections):
                if d_idx in det_matched:
                    continue
                iou = self._iou(track['bbox'], det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_iou >= self.iou_threshold and best_det_idx >= 0:
                det = detections[best_det_idx]
                prev_bbox = track['bbox']
                prev_centroid = ((prev_bbox[0] + prev_bbox[2]) / 2.0, (prev_bbox[1] + prev_bbox[3]) / 2.0)
                curr_centroid = ((det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0)

                self.tracks[tid]['bbox'] = det[:4]
                self.tracks[tid]['score'] = det[4] if len(det) > 4 else 1.0
                self.tracks[tid]['class_id'] = int(det[5]) if len(det) > 5 else int(track.get('class_id', 2))
                self.tracks[tid]['hits'] += 1
                self.tracks[tid]['age'] = 0
                self.tracks[tid]['prev_centroid'] = prev_centroid
                self.tracks[tid]['curr_centroid'] = curr_centroid
                matched.add(tid)
                det_matched.add(best_det_idx)

        # Increment age of unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        # Register new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in det_matched:
                centroid = ((det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0)
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'score': det[4] if len(det) > 4 else 1.0,
                    'class_id': int(det[5]) if len(det) > 5 else 2,
                    'hits': 1,
                    'age': 0,
                    'prev_centroid': centroid,
                    'curr_centroid': centroid,
                    'speed_kmh': 0.0
                }
                self.next_id += 1

        # Return confirmed tracks (active and verified)
        for tid, track in self.tracks.items():
            if track['hits'] >= self.min_hits or track['age'] == 0:
                b = track['bbox']
                updated_tracks.append([b[0], b[1], b[2], b[3], tid, track.get('class_id', 2)])

        return np.array(updated_tracks) if updated_tracks else np.empty((0, 6))

    def estimate_speed(self, track_id, ppm=8.0, fps=30.0):
        """
        Estimate speed in km/h based on centroid displacement over frame interval.
        """
        if track_id not in self.tracks:
            return 0.0

        track = self.tracks[track_id]
        p1 = track['prev_centroid']
        p2 = track['curr_centroid']

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_pixels = np.sqrt(dx ** 2 + dy ** 2)

        if dist_pixels < 1.0:
            return track.get('speed_kmh', 0.0)

        # meters = pixels / ppm
        meters = dist_pixels / max(1.0, ppm)
        speed_mps = meters * fps
        current_speed = speed_mps * 3.6  # convert m/s to km/h

        # Exponential moving average filter for smooth speed output
        prev_speed = track.get('speed_kmh', 0.0)
        smoothed = (0.7 * current_speed + 0.3 * prev_speed) if prev_speed > 0 else current_speed

        # Cap realistic bounds
        smoothed = min(180.0, max(0.0, smoothed))
        track['speed_kmh'] = smoothed
        return smoothed
