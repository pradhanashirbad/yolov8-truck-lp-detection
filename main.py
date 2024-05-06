from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="config.yaml", epochs=100)  # train the model

# metrics = model.val()

# results = model(source = 'data_1500/test/images', name='test_data_pt', save=True, exist_ok=True)