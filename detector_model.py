# infer_webcam_pt.py
from ultralytics import YOLO
import cv2
from pathlib import Path


MODEL_PATH = Path(__file__).parent /"runs" / "detect" / "wider_face_exp" / "weights" / "best.pt"
IMGSZ = 320
CONF = 0.25
IOU = 0.45

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for box in boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Detección (q para salir)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()