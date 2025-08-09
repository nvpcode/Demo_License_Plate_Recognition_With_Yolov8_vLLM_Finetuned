import streamlit as st
from ultralytics import YOLO
import cv2
import time
from PIL import Image
from unsloth import FastVisionModel
import numpy as np
import re
import threading

# ----------------------------
# Helper: load models once
# ----------------------------
@st.cache_resource
def load_models(yolo_path="files_model/license_plate_detector_yolov8.pt", unsloth_path="files_model/unsloth_finetune"):
    yolo = YOLO(yolo_path)
    ocr_model, ocr_tokenizer = FastVisionModel.from_pretrained(model_name=unsloth_path, load_in_4bit=True)
    FastVisionModel.for_inference(ocr_model)
    return yolo, ocr_model, ocr_tokenizer

# ----------------------------
# Recognizer class (same logic as yours)
# ----------------------------
class LicensePlateRecognizer:
    def __init__(self, yolo, ocr_model, ocr_tokenizer, device='cuda:0'):
        self.yolo = yolo
        self.ocr_model = ocr_model
        self.ocr_tokenizer = ocr_tokenizer
        self.device = device

    def detect_plates(self, image):
        # image: BGR numpy array
        results = self.yolo(image)[0]
        plates = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # ensure coordinates within image
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            plate_img = image[y1:y2, x1:x2]
            plates.append((plate_img, (x1, y1, x2, y2)))
        return plates

    def extract_text(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return ""
        image_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        instruction = (
            "You are a world-class OCR expert specializing in recognizing all types of vehicle license plates. "
            "Extract ONLY the exact license plate text using digits (0-9), uppercase letters (A-Z), hyphen (-), and dot (.)."
        )

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        input_text = self.ocr_tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.ocr_tokenizer(pil_image, input_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        outputs = self.ocr_model.generate(**inputs, max_new_tokens=128, temperature=1.0, min_p=0.1)
        output_text = self.ocr_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text.split("assistant")[-1].strip()

    def preprocess_plate_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip().upper()
        return re.sub(r'[^A-Z0-9\-.]', '', text)

# ----------------------------
# Video capture thread for RTSP / webcam
# ----------------------------
class VideoCaptureThread:
    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {self.src}")
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self.lock:
                self.frame = frame
        if self.cap:
            self.cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="LPR - Real-time", page_icon="🚘", layout="wide")
st.title("🚘 License Plate Recognition — Image & Real-time Stream")

# Load models once
with st.spinner("Loading models (YOLO + OCR)... this can take a while"):
    yolo_model, ocr_model, ocr_tokenizer = load_models()
    recognizer = LicensePlateRecognizer(yolo_model, ocr_model, ocr_tokenizer)

st.sidebar.header("Mode")
mode = st.sidebar.radio("Choose mode", ("Image Upload", "Webcam (local)", "RTSP / IP Camera"))

# common controls
display_fps = st.sidebar.checkbox("Show FPS", value=True)
show_boxes = st.sidebar.checkbox("Show bounding boxes & text", value=True)
max_boxes = st.sidebar.slider("Max plates to display per frame", 1, 10, 1)

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        plates = recognizer.detect_plates(image)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original image", use_container_width=True)
        with col2:
            if not plates:
                st.warning("No plates detected.")
            else:
                start = time.time()

                for i, (plate_img, (x1, y1, x2, y2)) in enumerate(plates[:max_boxes]):
                    text = recognizer.extract_text(plate_img)
                    text_clean = recognizer.preprocess_plate_text(text)

                    # Hiển thị ảnh
                    st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

                    # Hiển thị caption to, màu đỏ
                    st.markdown(
                        f"<h3 style='color:red; text-align:left;'>Plate #{i+1}: {text_clean}</h3>",
                        unsafe_allow_html=True
                    )
                
                elapsed = time.time() - start
                print('\n⏱️ Thời gian xử lý tổng: {:02d}:{:02d}:{:02d}'.format(
                    int(elapsed // 3600),
                    int((elapsed % 3600) // 60),
                    int(elapsed % 60)
                ))
                   
elif mode in ("Webcam (local)", "RTSP / IP Camera"):
    if mode == "Webcam (local)":
        src = st.sidebar.text_input("Webcam index", "0")
    else:
        src = st.sidebar.text_input("RTSP/HTTP URL", "rtsp://username:password@192.168.x.x:554/stream")

    start_button = st.button("Start Stream")
    stop_button = st.button("Stop Stream")

    # a place to show the video
    video_slot = st.empty()
    info_slot = st.empty()

    # keep capture object in session state to persist between runs
    if "video_thread" not in st.session_state:
        st.session_state.video_thread = None

    if start_button:
        # start thread
        try:
            source = int(src) if mode == "Webcam (local)" and str(src).isdigit() else src
            vt = VideoCaptureThread(source)
            vt.start()
            st.session_state.video_thread = vt
            info_slot.success("Streaming started")
        except Exception as e:
            st.session_state.video_thread = None
            info_slot.error(f"Failed to start stream: {e}")

    if stop_button and st.session_state.video_thread is not None:
        st.session_state.video_thread.stop()
        st.session_state.video_thread = None
        info_slot.info("Streaming stopped")

    # stream loop (Streamlit reruns script frequently, so keep lightweight)
    if st.session_state.video_thread is not None:
        last_time = time.time()
        fps = 0.0
        try:
            while st.session_state.video_thread is not None and st.session_state.video_thread.running:
                frame = st.session_state.video_thread.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                start_proc = time.time()
                plates = recognizer.detect_plates(frame)

                # draw boxes and OCR
                for i, (plate_img, (x1, y1, x2, y2)) in enumerate(plates[:max_boxes]):
                    text = recognizer.extract_text(plate_img)
                    text_clean = recognizer.preprocess_plate_text(text)
                    if show_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text_clean, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if display_fps:
                    now = time.time()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_time))
                    last_time = now
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # show frame
                video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                # break if user pressed stop (Streamlit re-runs, so check session state)
                time.sleep(0.03)
        except Exception as e:
            info_slot.error(f"Stream error: {e}")
            st.session_state.video_thread = None

# Footer / tips
st.markdown("---")
st.write("**Tips:**")
st.write("- For RTSP streams, use the camera's RTSP URL (often rtsp://user:pass@ip:554/...).")
st.write("- Use GPU (PyTorch + CUDA) for faster OCR and YOLO inference. If no GPU, reduce frame rate.")
st.write("- If the RTSP stream fails, try increasing the network timeout or checking credentials.")
