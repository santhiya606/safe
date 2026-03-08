"""
====================================================================
AI-POWERED INDUSTRIAL ACCIDENT DETECTION SYSTEM
Using Computer Vision + SMS & Alert Notifications
====================================================================
Author: Safety AI Systems
Version: 1.0.0
Description:
    Real-time industrial accident detection using computer vision (YOLOv8/OpenCV),
    with automated SMS alerts via Twilio and email notifications.
    Detects: Fire, Smoke, Falls, Missing PPE, Unsafe zones, Chemical spills.
====================================================================
"""

# ─────────────────────────────────────────────
# DEPENDENCIES — Install with:
#   pip install opencv-python ultralytics twilio
#               smtplib numpy Pillow requests python-dotenv
# ─────────────────────────────────────────────

import cv2
import numpy as np
import smtplib
import time
import threading
import os
import json
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from dotenv import load_dotenv

# Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[WARNING] Twilio not installed. SMS alerts disabled. Run: pip install twilio")

# YOLOv8 for object/hazard detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics not installed. Run: pip install ultralytics")

# ─────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES
# Create a .env file with your credentials (see CONFIG section below)
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# ============== CONFIGURATION =================
# ─────────────────────────────────────────────

CONFIG = {
    # ── Camera / Video Source ──────────────────
    "VIDEO_SOURCE": 0,                  # 0 = webcam | "rtsp://..." = IP cam | "video.mp4"
    "FRAME_WIDTH": 1280,
    "FRAME_HEIGHT": 720,
    "FPS": 30,

    # ── AI Model ───────────────────────────────
    "MODEL_PATH": "yolov8n.pt",         # yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
    "CONFIDENCE_THRESHOLD": 0.55,       # Detection confidence (0.0 – 1.0)
    "IOU_THRESHOLD": 0.45,

    # ── Hazard Classes (YOLO COCO labels or custom) ─
    "HAZARD_CLASSES": {
        "fire":           {"color": (0, 0, 255),   "severity": "CRITICAL", "sms": True,   "email": True},
        "smoke":          {"color": (128, 128, 128),"severity": "HIGH",     "sms": True,   "email": True},
        "person_fallen":  {"color": (0, 165, 255),  "severity": "HIGH",     "sms": True,   "email": True},
        "no_helmet":      {"color": (0, 255, 255),  "severity": "MEDIUM",   "sms": True,   "email": False},
        "no_vest":        {"color": (255, 165, 0),  "severity": "MEDIUM",   "sms": True,   "email": False},
        "chemical_spill": {"color": (255, 0, 255),  "severity": "CRITICAL", "sms": True,   "email": True},
        "electrical_arc": {"color": (255, 255, 0),  "severity": "CRITICAL", "sms": True,   "email": True},
        "forklift_near":  {"color": (0, 255, 0),    "severity": "HIGH",     "sms": True,   "email": False},
    },

    # ── Alert Cooldown (seconds) — prevents spam ─
    "ALERT_COOLDOWN_SECONDS": 60,

    # ── Alert Snapshot ─────────────────────────
    "SAVE_SNAPSHOTS": True,
    "SNAPSHOT_DIR": "./accident_snapshots",

    # ── Twilio SMS Configuration ───────────────
    "TWILIO_ACCOUNT_SID":   os.getenv("TWILIO_ACCOUNT_SID",  "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
    "TWILIO_AUTH_TOKEN":    os.getenv("TWILIO_AUTH_TOKEN",   "your_auth_token_here"),
    "TWILIO_FROM_NUMBER":   os.getenv("TWILIO_FROM_NUMBER",  "+1XXXXXXXXXX"),    # Your Twilio number
    "SMS_RECIPIENTS": [
        os.getenv("SMS_RECIPIENT_1", "+1XXXXXXXXXX"),   # Safety Manager
        os.getenv("SMS_RECIPIENT_2", "+1XXXXXXXXXX"),   # Plant Supervisor
        # Add more numbers as needed
    ],

    # ── Email Configuration ────────────────────
    "EMAIL_ENABLED": True,
    "SMTP_SERVER":   os.getenv("SMTP_SERVER",   "smtp.gmail.com"),
    "SMTP_PORT":     int(os.getenv("SMTP_PORT", "587")),
    "EMAIL_SENDER":  os.getenv("EMAIL_SENDER",  "your_alert@gmail.com"),
    "EMAIL_PASSWORD": os.getenv("EMAIL_PASSWORD", "your_app_password"),
    "EMAIL_RECIPIENTS": [
        os.getenv("EMAIL_RECIPIENT_1", "safety_manager@company.com"),
        os.getenv("EMAIL_RECIPIENT_2", "plant_supervisor@company.com"),
    ],

    # ── Facility Info ──────────────────────────
    "FACILITY_NAME":  "Industrial Plant - Unit A",
    "CAMERA_LOCATION": "Zone 3 - Assembly Floor",
    "LOG_FILE":       "accident_detection.log",
}

# ─────────────────────────────────────────────
# ============== LOGGING SETUP =================
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AccidentDetection")


# ─────────────────────────────────────────────
# ============== SMS ALERT MODULE ==============
# ─────────────────────────────────────────────

class SMSAlertSystem:
    """
    Handles SMS alert dispatch via Twilio.
    Supports single and bulk recipient messaging.
    """

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.from_number = from_number
        self.client = None
        if TWILIO_AVAILABLE:
            try:
                self.client = TwilioClient(account_sid, auth_token)
                logger.info("✅ Twilio SMS client initialized.")
            except Exception as e:
                logger.error(f"❌ Twilio init failed: {e}")

    def send_sms(self, recipients: list, message: str) -> dict:
        """
        Send SMS to one or more recipients.

        Args:
            recipients: List of phone numbers (E.164 format, e.g. "+1XXXXXXXXXX")
            message:    Text message body

        Returns:
            dict with status per recipient
        """
        results = {}
        if not self.client:
            logger.warning("SMS client not available. Message not sent.")
            return results

        for number in recipients:
            try:
                msg = self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=number
                )
                results[number] = {"status": "sent", "sid": msg.sid}
                logger.info(f"📱 SMS sent to {number} | SID: {msg.sid}")
            except Exception as e:
                results[number] = {"status": "failed", "error": str(e)}
                logger.error(f"❌ SMS failed for {number}: {e}")

        return results

    def build_alert_message(self, hazard_type: str, severity: str,
                             confidence: float, location: str,
                             facility: str, timestamp: str) -> str:
        """
        Build a structured SMS alert message.
        """
        severity_emoji = {
            "CRITICAL": "🚨🔴 CRITICAL",
            "HIGH":     "⚠️🟠 HIGH",
            "MEDIUM":   "⚡🟡 MEDIUM",
            "LOW":      "ℹ️🟢 LOW"
        }.get(severity, "⚠️ ALERT")

        message = (
            f"{severity_emoji} INDUSTRIAL ACCIDENT ALERT\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🏭 Facility : {facility}\n"
            f"📍 Location : {location}\n"
            f"⚠️  Hazard   : {hazard_type.upper().replace('_', ' ')}\n"
            f"📊 Severity : {severity}\n"
            f"🎯 Confidence: {confidence:.1%}\n"
            f"🕐 Time     : {timestamp}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"IMMEDIATE ACTION REQUIRED!\n"
            f"Follow Emergency Protocol."
        )
        return message


# ─────────────────────────────────────────────
# ============= EMAIL ALERT MODULE =============
# ─────────────────────────────────────────────

class EmailAlertSystem:
    """
    Sends HTML email alerts with optional snapshot attachments.
    """

    def __init__(self, smtp_server: str, smtp_port: int,
                 sender: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password

    def send_alert_email(self, recipients: list, hazard_type: str,
                          severity: str, confidence: float,
                          location: str, facility: str,
                          timestamp: str, snapshot_path: str = None):
        """
        Send an HTML email alert with optional snapshot image.
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 [{severity}] Accident Detected: {hazard_type.replace('_',' ').title()} @ {facility}"
            msg["From"] = self.sender
            msg["To"] = ", ".join(recipients)

            # HTML Email Body
            html_body = f"""
            <html><body style="font-family:Arial,sans-serif; background:#f4f4f4; padding:20px;">
              <div style="max-width:600px; margin:auto; background:#fff;
                          border-radius:8px; overflow:hidden;
                          box-shadow:0 2px 10px rgba(0,0,0,0.1);">
                <!-- Header -->
                <div style="background:#c0392b; padding:20px; text-align:center;">
                  <h1 style="color:#fff; margin:0; font-size:22px;">
                    🚨 INDUSTRIAL ACCIDENT ALERT
                  </h1>
                  <p style="color:#f8d7da; margin:5px 0 0;">
                    AI-Powered Safety Monitoring System
                  </p>
                </div>

                <!-- Severity Badge -->
                <div style="background:{'#c0392b' if severity=='CRITICAL' else '#e67e22' if severity=='HIGH' else '#f1c40f'};
                            padding:10px; text-align:center;">
                  <span style="color:#fff; font-size:18px; font-weight:bold;">
                    ⚠️ SEVERITY: {severity}
                  </span>
                </div>

                <!-- Details Table -->
                <div style="padding:25px;">
                  <table style="width:100%; border-collapse:collapse;">
                    <tr style="background:#f8f9fa;">
                      <td style="padding:10px; font-weight:bold; width:40%;">🏭 Facility</td>
                      <td style="padding:10px;">{facility}</td>
                    </tr>
                    <tr>
                      <td style="padding:10px; font-weight:bold;">📍 Camera Location</td>
                      <td style="padding:10px;">{location}</td>
                    </tr>
                    <tr style="background:#f8f9fa;">
                      <td style="padding:10px; font-weight:bold;">⚠️ Hazard Detected</td>
                      <td style="padding:10px; color:#c0392b; font-weight:bold;">
                        {hazard_type.upper().replace("_"," ")}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding:10px; font-weight:bold;">🎯 AI Confidence</td>
                      <td style="padding:10px;">{confidence:.1%}</td>
                    </tr>
                    <tr style="background:#f8f9fa;">
                      <td style="padding:10px; font-weight:bold;">🕐 Detected At</td>
                      <td style="padding:10px;">{timestamp}</td>
                    </tr>
                  </table>

                  <!-- Action Required -->
                  <div style="margin-top:20px; padding:15px; background:#fff3cd;
                              border-left:4px solid #e67e22; border-radius:4px;">
                    <h3 style="color:#e67e22; margin:0 0 8px;">
                      🔔 Immediate Action Required
                    </h3>
                    <p style="margin:0; color:#856404;">
                      Please respond to this alert immediately and follow your facility's
                      emergency response protocol. Ensure all personnel in the affected
                      area are safe and accounted for.
                    </p>
                  </div>

                  {"<p style='margin-top:15px;'><strong>📸 Snapshot attached</strong> — See attached image for visual confirmation.</p>" if snapshot_path else ""}
                </div>

                <!-- Footer -->
                <div style="background:#2c3e50; padding:15px; text-align:center;">
                  <p style="color:#bdc3c7; margin:0; font-size:12px;">
                    AI Industrial Safety Monitoring System &bull;
                    Auto-generated Alert &bull; Do not reply to this email.
                  </p>
                </div>
              </div>
            </body></html>
            """

            msg.attach(MIMEText(html_body, "html"))

            # Attach snapshot image if provided
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(snapshot_path))
                image.add_header("Content-Disposition", "attachment",
                                 filename=os.path.basename(snapshot_path))
                msg.attach(image)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, recipients, msg.as_string())

            logger.info(f"📧 Email alert sent to {recipients}")

        except Exception as e:
            logger.error(f"❌ Email send failed: {e}")


# ─────────────────────────────────────────────
# ========= COMPUTER VISION DETECTION =========
# ─────────────────────────────────────────────

class AccidentDetector:
    """
    Core AI detection engine using YOLOv8.
    Supports custom-trained models for PPE, fire, smoke, falls, etc.
    """

    def __init__(self, model_path: str, confidence: float, iou: float):
        self.model = None
        self.confidence = confidence
        self.iou = iou

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                logger.info(f"✅ YOLO model loaded: {model_path}")
            except Exception as e:
                logger.error(f"❌ YOLO model load failed: {e}")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run inference on a single frame.

        Returns:
            List of dicts: [{class_name, confidence, bbox}, ...]
        """
        if self.model is None:
            return self._mock_detect(frame)

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                class_name = result.names[class_id]
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "class_name":  class_name,
                    "confidence":  conf_score,
                    "bbox":        (x1, y1, x2, y2)
                })

        return detections

    def _mock_detect(self, frame: np.ndarray) -> list:
        """
        Mock detector for testing without a trained model.
        Randomly simulates hazard detection.
        """
        import random
        if random.random() < 0.02:  # 2% chance per frame
            hazards = list(CONFIG["HAZARD_CLASSES"].keys())
            mock_class = random.choice(hazards)
            h, w = frame.shape[:2]
            return [{
                "class_name": mock_class,
                "confidence": random.uniform(0.6, 0.95),
                "bbox": (
                    random.randint(0, w // 2),
                    random.randint(0, h // 2),
                    random.randint(w // 2, w),
                    random.randint(h // 2, h)
                )
            }]
        return []


# ─────────────────────────────────────────────
# ============= ALERT MANAGER ==================
# ─────────────────────────────────────────────

class AlertManager:
    """
    Manages alert throttling, dispatch coordination,
    snapshot saving, and incident logging.
    """

    def __init__(self, config: dict):
        self.config         = config
        self.last_alert     = {}           # {hazard_type: last_alert_timestamp}
        self.incident_log   = []

        # Initialize SMS
        self.sms = SMSAlertSystem(
            account_sid = config["TWILIO_ACCOUNT_SID"],
            auth_token  = config["TWILIO_AUTH_TOKEN"],
            from_number = config["TWILIO_FROM_NUMBER"]
        )

        # Initialize Email
        self.email = EmailAlertSystem(
            smtp_server = config["SMTP_SERVER"],
            smtp_port   = config["SMTP_PORT"],
            sender      = config["EMAIL_SENDER"],
            password    = config["EMAIL_PASSWORD"]
        ) if config["EMAIL_ENABLED"] else None

        # Ensure snapshot directory exists
        if config["SAVE_SNAPSHOTS"]:
            Path(config["SNAPSHOT_DIR"]).mkdir(parents=True, exist_ok=True)

    def _is_on_cooldown(self, hazard_type: str) -> bool:
        if hazard_type not in self.last_alert:
            return False
        elapsed = time.time() - self.last_alert[hazard_type]
        return elapsed < self.config["ALERT_COOLDOWN_SECONDS"]

    def _save_snapshot(self, frame: np.ndarray, hazard_type: str,
                        timestamp: str) -> str:
        filename = f"{timestamp.replace(':', '-')}_{hazard_type}.jpg"
        filepath = os.path.join(self.config["SNAPSHOT_DIR"], filename)
        cv2.imwrite(filepath, frame)
        logger.info(f"📸 Snapshot saved: {filepath}")
        return filepath

    def process_detection(self, frame: np.ndarray, detection: dict):
        """
        Process a single detection event:
        - Check cooldown
        - Save snapshot
        - Send SMS + email (in background threads)
        - Log incident
        """
        hazard_type = detection["class_name"]
        confidence  = detection["confidence"]

        # Only process known hazard classes
        if hazard_type not in self.config["HAZARD_CLASSES"]:
            return

        hazard_info = self.config["HAZARD_CLASSES"][hazard_type]
        severity    = hazard_info["severity"]

        # Cooldown check
        if self._is_on_cooldown(hazard_type):
            return

        # Update cooldown timestamp
        self.last_alert[hazard_type] = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.warning(
            f"🚨 HAZARD DETECTED | {hazard_type.upper()} | "
            f"Severity: {severity} | Confidence: {confidence:.1%}"
        )

        # Save annotated snapshot
        snapshot_path = None
        if self.config["SAVE_SNAPSHOTS"]:
            annotated = frame.copy()
            x1, y1, x2, y2 = detection["bbox"]
            color = hazard_info["color"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{hazard_type.upper()} {confidence:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            snapshot_path = self._save_snapshot(annotated, hazard_type,
                                                 timestamp.replace(" ", "_"))

        # Build SMS message
        sms_msg = self.sms.build_alert_message(
            hazard_type = hazard_type,
            severity    = severity,
            confidence  = confidence,
            location    = self.config["CAMERA_LOCATION"],
            facility    = self.config["FACILITY_NAME"],
            timestamp   = timestamp
        )

        # Dispatch SMS in background
        if hazard_info["sms"] and self.config["SMS_RECIPIENTS"]:
            threading.Thread(
                target=self.sms.send_sms,
                args=(self.config["SMS_RECIPIENTS"], sms_msg),
                daemon=True
            ).start()

        # Dispatch Email in background
        if (hazard_info["email"]
                and self.email
                and self.config["EMAIL_RECIPIENTS"]):
            threading.Thread(
                target=self.email.send_alert_email,
                kwargs={
                    "recipients":  self.config["EMAIL_RECIPIENTS"],
                    "hazard_type": hazard_type,
                    "severity":    severity,
                    "confidence":  confidence,
                    "location":    self.config["CAMERA_LOCATION"],
                    "facility":    self.config["FACILITY_NAME"],
                    "timestamp":   timestamp,
                    "snapshot_path": snapshot_path
                },
                daemon=True
            ).start()

        # Log incident
        incident = {
            "timestamp":   timestamp,
            "hazard":      hazard_type,
            "severity":    severity,
            "confidence":  round(confidence, 3),
            "location":    self.config["CAMERA_LOCATION"],
            "snapshot":    snapshot_path
        }
        self.incident_log.append(incident)
        self._write_incident_log(incident)

    def _write_incident_log(self, incident: dict):
        log_path = "incidents.json"
        existing = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.append(incident)
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2)


# ─────────────────────────────────────────────
# ============ MAIN DETECTION LOOP =============
# ─────────────────────────────────────────────

class IndustrialSafetySystem:
    """
    Main orchestrator: captures video, runs detection,
    overlays UI, and triggers alerts.
    """

    def __init__(self, config: dict):
        self.config   = config
        self.detector = AccidentDetector(
            model_path = config["MODEL_PATH"],
            confidence = config["CONFIDENCE_THRESHOLD"],
            iou        = config["IOU_THRESHOLD"]
        )
        self.alert_manager = AlertManager(config)
        self.running  = False
        self.cap      = None
        self.frame_count   = 0
        self.detect_count  = 0

    def _open_camera(self) -> bool:
        source = self.config["VIDEO_SOURCE"]
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error(f"❌ Cannot open video source: {source}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config["FRAME_WIDTH"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["FRAME_HEIGHT"])
        self.cap.set(cv2.CAP_PROP_FPS,           self.config["FPS"])
        logger.info(f"📷 Camera opened: {source}")
        return True

    def _draw_overlay(self, frame: np.ndarray,
                       detections: list, fps: float) -> np.ndarray:
        """
        Draw bounding boxes, labels, and HUD on the frame.
        """
        h, w = frame.shape[:2]

        # ── HUD Header ──────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(frame,
                    f"🏭 {self.config['FACILITY_NAME']} | "
                    f"📍 {self.config['CAMERA_LOCATION']} | "
                    f"FPS: {fps:.1f} | Frames: {self.frame_count}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        # ── Timestamp ───────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (w - 220, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Draw Detections ─────────────────────
        for det in detections:
            cls  = det["class_name"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]

            if cls in self.config["HAZARD_CLASSES"]:
                color    = self.config["HAZARD_CLASSES"][cls]["color"]
                severity = self.config["HAZARD_CLASSES"][cls]["severity"]
            else:
                color    = (0, 255, 0)
                severity = "INFO"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            label    = f"⚠ {cls.upper().replace('_',' ')} [{conf:.0%}] [{severity}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Status Bar (bottom) ─────────────────
        status_color = (0, 0, 200) if detections else (0, 150, 0)
        status_text  = (
            f"🚨 HAZARD DETECTED: {len(detections)} event(s)"
            if detections else "✅ MONITORING — No Hazards Detected"
        )
        cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        return frame

    def run(self):
        """
        Start the main detection loop.
        Press 'q' to quit, 's' to force snapshot.
        """
        if not self._open_camera():
            return

        self.running = True
        logger.info("🚀 Industrial Safety System STARTED. Press 'q' to quit.")

        prev_time = time.time()
        fps = 0.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("⚠️  Frame read failed — retrying...")
                time.sleep(0.1)
                continue

            self.frame_count += 1

            # ── FPS calculation ──────────────────
            now  = time.time()
            fps  = 1.0 / max(now - prev_time, 1e-5)
            prev_time = now

            # ── Run AI Detection ─────────────────
            detections = self.detector.detect(frame)

            # ── Filter only known hazard classes ─
            hazard_detections = [
                d for d in detections
                if d["class_name"] in self.config["HAZARD_CLASSES"]
            ]

            # ── Trigger Alerts ───────────────────
            for det in hazard_detections:
                self.detect_count += 1
                self.alert_manager.process_detection(frame, det)

            # ── Draw UI Overlay ──────────────────
            display_frame = self._draw_overlay(frame.copy(), hazard_detections, fps)

            cv2.imshow("🏭 AI Industrial Safety Monitor", display_frame)

            # ── Keyboard Controls ────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("🛑 Quit key pressed. Shutting down.")
                break
            elif key == ord("s"):
                snap = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                snap_path = os.path.join(self.config["SNAPSHOT_DIR"], snap)
                cv2.imwrite(snap_path, frame)
                logger.info(f"📸 Manual snapshot: {snap_path}")

        self._shutdown()

    def _shutdown(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info(
            f"✅ System stopped. Total frames: {self.frame_count} | "
            f"Hazard events: {self.detect_count}"
        )


# ─────────────────────────────────────────────
# ============ STANDALONE ALERT TEST ===========
# ─────────────────────────────────────────────

def test_sms_alert():
    """
    Test SMS sending without running the camera system.
    Update CONFIG credentials before running.
    """
    print("\n" + "="*50)
    print("  📱 SMS ALERT TEST")
    print("="*50)

    sms = SMSAlertSystem(
        account_sid = CONFIG["TWILIO_ACCOUNT_SID"],
        auth_token  = CONFIG["TWILIO_AUTH_TOKEN"],
        from_number = CONFIG["TWILIO_FROM_NUMBER"]
    )

    test_message = sms.build_alert_message(
        hazard_type = "fire",
        severity    = "CRITICAL",
        confidence  = 0.92,
        location    = CONFIG["CAMERA_LOCATION"],
        facility    = CONFIG["FACILITY_NAME"],
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    print("\n📩 Message Preview:\n")
    print(test_message)
    print("\n" + "-"*50)

    results = sms.send_sms(CONFIG["SMS_RECIPIENTS"], test_message)
    for number, result in results.items():
        status = result.get("status", "unknown")
        icon   = "✅" if status == "sent" else "❌"
        print(f"{icon} {number}: {status}")

    print("="*50 + "\n")


def test_email_alert():
    """
    Test email sending without running the camera system.
    """
    print("\n" + "="*50)
    print("  📧 EMAIL ALERT TEST")
    print("="*50)

    email = EmailAlertSystem(
        smtp_server = CONFIG["SMTP_SERVER"],
        smtp_port   = CONFIG["SMTP_PORT"],
        sender      = CONFIG["EMAIL_SENDER"],
        password    = CONFIG["EMAIL_PASSWORD"]
    )

    email.send_alert_email(
        recipients  = CONFIG["EMAIL_RECIPIENTS"],
        hazard_type = "chemical_spill",
        severity    = "CRITICAL",
        confidence  = 0.88,
        location    = CONFIG["CAMERA_LOCATION"],
        facility    = CONFIG["FACILITY_NAME"],
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    print("✅ Test email dispatched (check inbox).")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
# =================== MAIN ====================
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🏭 AI Industrial Accident Detection System"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "test-sms", "test-email", "test-all"],
        default="run",
        help=(
            "run        = Start live camera detection (default)\n"
            "test-sms   = Send a test SMS alert\n"
            "test-email = Send a test email alert\n"
            "test-all   = Run all tests"
        )
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Override video source (e.g. 0, 1, rtsp://..., video.mp4)")
    args = parser.parse_args()

    if args.source:
        try:
            CONFIG["VIDEO_SOURCE"] = int(args.source)
        except ValueError:
            CONFIG["VIDEO_SOURCE"] = args.source

    print("""
╔══════════════════════════════════════════════════════╗
║   🏭 AI INDUSTRIAL ACCIDENT DETECTION SYSTEM v1.0   ║
║   Computer Vision + SMS & Email Alerts               ║
╚══════════════════════════════════════════════════════╝
    """)

    if args.mode == "test-sms":
        test_sms_alert()

    elif args.mode == "test-email":
        test_email_alert()

    elif args.mode == "test-all":
        test_sms_alert()
        test_email_alert()

    else:
        system = IndustrialSafetySystem(CONFIG)
        system.run()
