"""
╔══════════════════════════════════════════════════════════════╗
║         FACE DETECTION SYSTEM — Animated PyQt6 GUI           ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import time
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QLabel, QHBoxLayout, QFrame, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QShortcut, QKeySequence, QIcon

# ── OpenCV Settings ──
CAMERA_INDEX  = 0
SAVE_FOLDER   = "saved_faces"
CASCADE_FILE  = "haarcascade_frontalface_default.xml"
SCALE_FACTOR  = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

COLOR_BOX   = (0, 255, 0)
COLOR_TEXT  = (255, 255, 255)
COLOR_BG    = (0, 0, 0)
COLOR_TITLE = (0, 200, 255)
COLOR_ALERT = (0, 0, 255)

# ── OpenCV Logic ──
def load_cascade():
    if os.path.exists(CASCADE_FILE):
        cascade_path = CASCADE_FILE
    else:
        cascade_path = os.path.join(cv2.data.haarcascades, CASCADE_FILE)
    if not os.path.exists(cascade_path):
        return None
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        return None
    return classifier

def detect_faces(frame, classifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = classifier.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, 
                                        minNeighbors=MIN_NEIGHBORS, minSize=MIN_FACE_SIZE, 
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    return (faces, gray) if len(faces) > 0 else ([], gray)

def draw_annotations(frame, faces):
    face_count = len(faces)
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BOX, 2)
        label = f"Face {i + 1}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - text_h - 8), (x + text_w + 4, y), COLOR_BOX, -1)
        cv2.putText(frame, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BG, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (30, 30, 30), -1)
    cv2.putText(frame, "Face Detection System", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TITLE, 2, cv2.LINE_AA)
    count_text = f"Faces: {face_count}"
    (cw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, count_text, (frame.shape[1] - cw - 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    h_frame = frame.shape[0]
    cv2.rectangle(frame, (0, h_frame - 32), (frame.shape[1], h_frame), (30, 30, 30), -1)
    cv2.putText(frame, "Press 'S' = Save  |  'Q' or ESC = Quit", (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return frame

def save_faces(frame, faces, folder=SAVE_FOLDER):
    if not os.path.exists(folder): os.makedirs(folder)
    timestamp = int(time.time())
    for i, (x, y, w, h) in enumerate(faces):
        margin = 20
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        cv2.imwrite(os.path.join(folder, f"face_{timestamp}_{i+1}.jpg"), frame[y1:y2, x1:x2])

# ── Themes ──
THEMES = {
    "Cosmic": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e1b4b, stop:0.5 #581c87, stop:1 #9d174d)",
    "Ocean": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f172a, stop:0.5 #155e75, stop:1 #1e3a8a)",
    "Sunset": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #111827, stop:0.5 #7c2d12, stop:1 #7f1d1d)",
    "Cyberpunk": "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #022c22, stop:0.5 #064e3b, stop:1 #0f172a)"
}

# ── PyQt6 Main Window ──
class FaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.classifier = load_cascade()
        if not self.classifier:
            QMessageBox.critical(self, "Error", f"Missing {CASCADE_FILE} in folder!")
            sys.exit(1)

        self.setWindowTitle("AI Face Scanner")
        self.setFixedSize(500, 550)
        self.current_theme = "Cosmic"
        
        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.apply_theme()

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Theme Selector (Top Right)
        self.theme_layout = QHBoxLayout()
        self.theme_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
        for name, gradient in THEMES.items():
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setToolTip(name)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"background: {gradient}; border-radius: 12px; border: 2px solid rgba(255,255,255,100);")
            btn.clicked.connect(lambda checked, n=name: self.change_theme(n))
            self.theme_layout.addWidget(btn)
        
        # Glassmorphism Container
        self.glass_frame = QFrame()
        self.glass_frame.setFixedSize(400, 420)
        self.glass_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 120);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 40);
            }
        """)
        
        # Inside the Glass Frame
        self.frame_layout = QVBoxLayout(self.glass_frame)
        self.frame_layout.setContentsMargins(30, 30, 30, 30)
        self.frame_layout.setSpacing(15)

        title = QLabel("Face Scanner")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("AI-Powered Detection System")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 170); background: transparent; border: none;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.frame_layout.addWidget(title)
        self.frame_layout.addWidget(subtitle)
        self.frame_layout.addSpacing(20)

        # Styled Buttons
        self.btn_webcam = self.create_button("Run Webcam Detection", "1", self.run_webcam)
        self.btn_image = self.create_button("Detect in Image File", "2", self.run_image)
        self.btn_exit = self.create_button("Exit System", "3", self.close, is_danger=True)

        self.layout.addLayout(self.theme_layout)
        self.layout.addWidget(self.glass_frame)

        # Keyboard Shortcuts
        QShortcut(QKeySequence("1"), self).activated.connect(self.run_webcam)
        QShortcut(QKeySequence("2"), self).activated.connect(self.run_image)
        QShortcut(QKeySequence("3"), self).activated.connect(self.close)

        # Sliding Animation on Startup
        self.anim = QPropertyAnimation(self.glass_frame, b"pos")
        self.anim.setDuration(800)
        self.anim.setStartValue(QPoint(50, 600))  # Start below the window
        self.anim.setEndValue(QPoint(50, 65))     # Slide to center
        self.anim.setEasingCurve(QEasingCurve.Type.OutBack) # Bouncy elastic effect
        self.anim.start()

    def create_button(self, text, shortcut, callback, is_danger=False):
        btn = QPushButton(f"  {text}      [Key: {shortcut}]")
        btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(55)
        
        base_color = "rgba(220, 38, 38, " if is_danger else "rgba(255, 255, 255, "
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {base_color} 20);
                color: white;
                border-radius: 12px;
                border: 1px solid {base_color} 50);
                text-align: left;
                padding-left: 15px;
            }}
            QPushButton:hover {{
                background-color: {base_color} 40);
                border: 1px solid {base_color} 100);
            }}
            QPushButton:pressed {{
                background-color: {base_color} 60);
            }}
        """)
        btn.clicked.connect(callback)
        self.frame_layout.addWidget(btn)
        return btn

    def change_theme(self, name):
        self.current_theme = name
        self.apply_theme()

    def apply_theme(self):
        self.central_widget.setStyleSheet(f"QWidget#central {{ background: {THEMES[self.current_theme]}; }}")
        self.central_widget.setObjectName("central")

    # ── Actions ──
    def run_webcam(self):
        self.hide() # Hide UI while OpenCV runs
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        paused, last_faces, last_frame = False, [], None
        while cap.isOpened():
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            if key == ord('p'): paused = not paused
            if key == ord('s') and last_frame is not None: save_faces(last_frame, last_faces)

            if not paused:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                faces, _ = detect_faces(frame, self.classifier)
                last_faces, last_frame = faces, frame.copy()
                disp = draw_annotations(frame, faces)
            else:
                disp = draw_annotations(last_frame.copy(), last_faces)
                cv2.putText(disp, " PAUSED ", (disp.shape[1]//2 - 80, disp.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_ALERT, 3)
            
            cv2.imshow("Webcam - Live (Press Q to quit)", disp)
            
        cap.release()
        cv2.destroyAllWindows()
        self.show() # Show UI again

    def run_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png *.jpeg)")
        if not file_path: return
        
        self.hide()
        frame = cv2.imread(file_path)
        if frame is not None:
            faces, _ = detect_faces(frame, self.classifier)
            annotated = draw_annotations(frame, faces)
            cv2.imshow("Image Detection - ANY KEY closes", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            out_path = "detected_" + os.path.basename(file_path)
            cv2.imwrite(out_path, annotated)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec())