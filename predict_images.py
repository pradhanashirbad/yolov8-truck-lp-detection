from ultralytics import YOLO


model_path = '/home/ashirbad/Documents/EAIGLE/truck_lp_yolov8/runs/detect/train3/weights/best.pt'
model = YOLO(model_path)
# model = YOLO("yolov8n.pt")

results = model(source = 'data_2000/test/images', name='2000_ep75_test_results_pt_200', save=True, exist_ok=True)