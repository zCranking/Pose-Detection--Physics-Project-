import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pandas as pd

class PosePhysics:
    LANDMARKS = {
        'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
        'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
        'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
        'LEFT_HIP': 23, 'RIGHT_HIP': 24,
        'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
        'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28
    }
    SEGMENT_MASS = {
        'upper_arm': 0.027, 'forearm': 0.016,
        'thigh': 0.10, 'shank': 0.046, 'torso': 0.50
    }
    SEGMENTS = [
        ('R_forearm', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('L_forearm', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
        ('R_upperarm', 'RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('L_upperarm', 'LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('R_thigh', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
        ('L_thigh', 'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
        ('R_shank', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
        ('L_shank', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
        ('torso', 'LEFT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP')
    ]
    SEGMENT_OFFSETS = {
        'R_forearm': (180, -80), 'L_forearm': (-180, -80),
        'R_upperarm': (180, 0), 'L_upperarm': (-180, 0),
        'R_thigh': (180, 120), 'L_thigh': (-180, 120),
        'R_shank': (180, 220), 'L_shank': (-180, 220),
        'torso': (0, -150)
    }

    def __init__(self, total_mass_kg=72.57, gravity=9.81, log_file='pose_metrics.csv', log_interval=30):
        self.total_mass = total_mass_kg
        self.gravity = gravity
        self.prev_lengths = {}
        self.prev_angles = {}
        self.prev_time = time.time()
        self.metrics = {}
        self.averages = {}
        self.frame_count = 0
        self.log_file = log_file
        self.log_interval = log_interval
        self.df = pd.DataFrame()

    def estimate_px_to_m(self, lmList):
        try:
            r = np.array(lmList[self.LANDMARKS['RIGHT_SHOULDER']][1:])
            l = np.array(lmList[self.LANDMARKS['LEFT_SHOULDER']][1:])
            px_dist = np.linalg.norm(r - l)
            return 0.41 / px_dist if px_dist > 0 else 0.001
        except Exception:
            return 0.001

    def get_segment_type(self, p1, p2, p3):
        if (p1, p2, p3) in [
            (self.LANDMARKS['RIGHT_SHOULDER'], self.LANDMARKS['RIGHT_ELBOW'], self.LANDMARKS['RIGHT_WRIST']),
            (self.LANDMARKS['LEFT_SHOULDER'], self.LANDMARKS['LEFT_ELBOW'], self.LANDMARKS['LEFT_WRIST'])
        ]:
            return 'forearm'
        if (p1, p2, p3) in [
            (self.LANDMARKS['RIGHT_HIP'], self.LANDMARKS['RIGHT_SHOULDER'], self.LANDMARKS['RIGHT_ELBOW']),
            (self.LANDMARKS['LEFT_HIP'], self.LANDMARKS['LEFT_SHOULDER'], self.LANDMARKS['LEFT_ELBOW'])
        ]:
            return 'upper_arm'
        if (p1, p2, p3) in [
            (self.LANDMARKS['RIGHT_SHOULDER'], self.LANDMARKS['RIGHT_HIP'], self.LANDMARKS['RIGHT_KNEE']),
            (self.LANDMARKS['LEFT_SHOULDER'], self.LANDMARKS['LEFT_HIP'], self.LANDMARKS['LEFT_KNEE'])
        ]:
            return 'thigh'
        if (p1, p2, p3) in [
            (self.LANDMARKS['RIGHT_HIP'], self.LANDMARKS['RIGHT_KNEE'], self.LANDMARKS['RIGHT_ANKLE']),
            (self.LANDMARKS['LEFT_HIP'], self.LANDMARKS['LEFT_KNEE'], self.LANDMARKS['LEFT_ANKLE'])
        ]:
            return 'shank'
        if (p1, p2, p3) in [
            (self.LANDMARKS['LEFT_SHOULDER'], self.LANDMARKS['LEFT_HIP'], self.LANDMARKS['RIGHT_HIP']),
            (self.LANDMARKS['RIGHT_SHOULDER'], self.LANDMARKS['RIGHT_HIP'], self.LANDMARKS['LEFT_HIP'])
        ]:
            return 'torso'
        return 'unknown'

    def calculate_metrics(self, lmList, p1, p2, p3, px_to_m=0.001, draw_img=None, label=''):
        if len(lmList) <= max(p1, p2, p3):
            return None

        pt1 = np.array(lmList[p1][1:])
        pt2 = np.array(lmList[p2][1:])
        pt3 = np.array(lmList[p3][1:])

        d1 = np.linalg.norm(pt1 - pt2) * px_to_m
        d2 = np.linalg.norm(pt3 - pt2) * px_to_m
        d3 = np.linalg.norm(pt1 - pt3) * px_to_m

        try:
            angle_rad = math.acos((d1**2 + d2**2 - d3**2) / (2 * d1 * d2))
            angle_deg = math.degrees(angle_rad)
        except (ValueError, ZeroDivisionError):
            angle_rad = 0.0
            angle_deg = 0.0

        segment_type = self.get_segment_type(p1, p2, p3)
        segment_mass = self.total_mass * self.SEGMENT_MASS.get(segment_type, 0.05)
        torque = segment_mass * self.gravity * (d2 / 2) * math.sin(angle_rad)
        cur_time = time.time()
        dt = cur_time - self.prev_time if self.prev_time else 1/30
        prev_len = self.prev_lengths.get((p1, p2, p3), d2)
        velocity = (d2 - prev_len) / dt if dt > 0 else 0

        self.prev_lengths[(p1, p2, p3)] = d2
        self.prev_time = cur_time
        self.prev_angles[(p1, p2, p3)] = angle_rad

        mid = tuple(np.mean([pt1, pt2, pt3], axis=0).astype(int))
        offset = self.SEGMENT_OFFSETS.get(label, (0, 0))
        pos = (mid[0] + offset[0], mid[1] + offset[1])

        metric = {
            'label': label,
            'segment': segment_type,
            'angle_deg': round(angle_deg, 1),
            'velocity_m_s': round(velocity, 2),
            'torque_Nm': round(torque, 2),
            'pos': pos
        }
        self.metrics[label] = metric
        if draw_img is not None:
            self.display_metrics(draw_img, metric)
        return metric

    def update_averages(self):
        for label, m in self.metrics.items():
            if label not in self.averages:
                self.averages[label] = {'angle_deg': [], 'velocity_m_s': [], 'torque_Nm': []}
            for stat in ['angle_deg', 'velocity_m_s', 'torque_Nm']:
                self.averages[label][stat].append(m[stat])
        for label in self.averages:
            for stat in self.averages[label]:
                if len(self.averages[label][stat]) > 100:
                    self.averages[label][stat] = self.averages[label][stat][-100:]

    def log_averages(self):
        row = {}
        for label, stats in self.averages.items():
            for stat, values in stats.items():
                if values:
                    row[f"{label}_{stat}"] = np.mean(values)
        if row:
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
            self.df.to_csv(self.log_file, index=False)

    def print_averages(self):
        for label, stats in self.averages.items():
            print(f"--- {label} ---")
            for stat, values in stats.items():
                if values:
                    print(f"{stat}: {np.mean(values):.2f}")
            print()

    def display_metrics(self, img, metric):
        x, y = metric['pos']
        lines = [
            f"{metric['label']} {metric['segment']}",
            f"Angle: {metric['angle_deg']}°",
            f"Vel: {metric['velocity_m_s']} m/s",
            f"Torque: {metric['torque_Nm']} Nm"
        ]
        for i, txt in enumerate(lines):
            cv2.putText(img, txt, (x+10, y+35*i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)

class poseDetector():
    def __init__(self, mode=False, model_complexity=1, segm=False, smooth=True, smooth_seg=True,
                 detectionCon=0.5, trackCon=0.5):
        self.metrics = {}
        self.mode = mode
        self.segm = segm
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.smooth_seg = smooth_seg
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.segm, self.smooth, self.smooth_seg,
                                     self.detectionCon, self.trackCon)
        self.lmList = []

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)
    pTime = 0
    detector = poseDetector()
    physics = PosePhysics(total_mass_kg=72.57)
    while True:
        success, img = cap.read()
        if not success:
            print("Could not read from webcam. Is it in use by another program?")
            break
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) > 28:
            px_to_m = physics.estimate_px_to_m(lmList)
            for label, p1, p2, p3 in PosePhysics.SEGMENTS:
                physics.calculate_metrics(
                    lmList,
                    PosePhysics.LANDMARKS[p1],
                    PosePhysics.LANDMARKS[p2],
                    PosePhysics.LANDMARKS[p3],
                    px_to_m=px_to_m,
                    draw_img=img,
                    label=label
                )
            physics.update_averages()
            if physics.frame_count % physics.log_interval == 0:
                physics.log_averages()
            physics.frame_count += 1
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (1700, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.resizeWindow("Image", 1920, 1080)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal running averages:")
    physics.print_averages()

if __name__ == "__main__":
    main()

#MAKE SURE TO TRY AND MAKE A SEPERATE FUNCITON THAT YOU CAN CALL THAT DOESN'T USE ANY PARAMETERS TO PRINT THE RETURN STUFF