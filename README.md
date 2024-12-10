# Fall-Detection
- FallDetection.ipynb: yolo class for training and inference, metrics (mAP50-95, mAR50-95, RAM CPU, RAM GPU, infer speed)
- data_augmentation.ipynb: read original dataset, augment data with changes of Brightness, Contrast, RandomGamma, ISONoise, Hue-Saturation Value, Rotate..., save augmented dataset for model training
- ml_backend: Label Studio ML backend integration
  - _wsgi.py: server for Label Studio ML backend
  - config.py: configuration for LS server, model choice, fps...
  - labeling_config.ls: label configuration for Label Studio
  - model.py: class to connect to Label Studio to get data and make predictions
  - video_analyzing.py: class to perform video analysis 

## Label Studio Integration

### Prerequisites

- Python 3.8+
- Label Studio
- label-studio-ml-backend

### Installation

1. Install Label Studio:
```bash
pip install label-studio
```

2. Install label-studio-ml-backend:
```bash
git clone https://github.com/yourusername/Fall-Detection
cd Fall-Detection/ml_backend
pip install -e .
```

### Server Setup

1. Launch Label Studio:
```bash
label-studio start
```

2. Launch label-studio-ml-backend:
```bash
label-studio-ml start ml_backend
```

### Connecting Servers

1. Create new Label Studio project
2. Navigate to Settings -> Machine Learning
3. Configure ML Backend:
- Title: Fall Detection Model
- URL: http://localhost:9090
4. Click Connect

### Using the System
1. Project Creation:
- Create new project
- Select Image Object Detection template
2. Image Upload:
- Click Import
- Choose upload method
- Select images
3. Prediction:
- Enable Auto-Annotation
- Upload images
- Click an image to get predictions

