import os
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
from mem0 import Memory

# 1. Define the Offline Configuration
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:1b",
            "base_url": "http://localhost:11434"
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"}
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "jarvis_memories",
            "path": "./chroma_db"
        }
    }
}

# 2. Initialize Memory with the local config
m = Memory.from_config(config)
print("🧠 Jarvis Mind initialized OFFLINE. No API Key required.")

# 3. LOAD BOTH BRAINS (General + Cigar)
general_model = YOLO('yolov8n.pt')
cigar_model = YOLO('runs/detect/jarvis_cigar_model/weights/best.pt')


def run_jarvis_ultra():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    last_memory_time = 0

    print("🧠 Jarvis 'Temporal Mind' is active. Monitoring starting...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run both AI models
        gen_res = general_model.predict(source=frame, device='cuda', verbose=False)
        cigar_res = cigar_model.predict(source=frame, device='cuda', verbose=False)

        current_time = time.time()
        annotated_frame = gen_res[0].plot()

        # Check for Cigar Detections
        if len(cigar_res[0].boxes) > 0:
            # Only record a memory once every 5 minutes to avoid clutter
            if current_time - last_memory_time > 300:
                mem_data = "Ethan is having a cigar session. It is currently " + time.ctime()
                m.add(mem_data, user_id="ethan_aguilar")
                print(f"📓 Memory Saved: {mem_data}")
                last_memory_time = current_time

        # Draw Cigar boxes manually
        for box in cigar_res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(annotated_frame, "CIGAR", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Jarvis Ultra-Monitor", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_jarvis_ultra()