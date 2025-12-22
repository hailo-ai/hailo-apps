# YOLOv8 Retraining Example

This guide demonstrates how to retrain a YOLOv8 model for barcode detection using the Kaggle barcode-detector dataset. After training, we'll convert the model to HEF format and deploy it on the Raspberry Pi 5 AI Kit (or any x86 platform with a Hailo accelerator).

#### Required compute resources:
- **Training phase**: Any cloud service with GPU resources (including Python notebook-based services like Google Colab)
- **Hailo compilation phase**: Conversion to HEF format is compute-intensive but manageable on standard PCs (e.g., overnight task)

For complete reference implementations, see the Jupyter notebooks (please note some of the stages, including installations, might take time): 
- [`retraining.ipynb`](retraining.ipynb)
- [`compilation.ipynb`](compilation.ipynb)

## Training Setup

### Set up environment

First, create a Python virtual environment:

```bash
python -m venv env
source env/bin/activate
```

### Download the dataset
 
Install the Kaggle dataset downloader:

```bash
pip install kagglehub
```

Visit the [barcode-detector](https://www.kaggle.com/datasets/kushagrapandya/barcode-detection) dataset page on Kaggle. Click "Download" and copy the Python code into a new script on your development machine:

```python
import kagglehub
path = kagglehub.dataset_download("kushagrapandya/barcode-detection")
print("Path to dataset files:", path)
```

Execute the script. The dataset will download to a location similar to: `~/.cache/kagglehub/datasets/kushagrapandya/barcode-detection/versions/1`. 

Examine the folder structure:
- Three subsets: `test`, `train`, and `valid`
- Each contains `images` and `labels` folders with corresponding filenames
- Label format: one object per row, starting with class number (0=Barcode, 1=QR Code) followed by bounding box coordinates

### Train the model

This process takes several hours. For quick validation, use `epochs=1` before running full-scale training.

Install dependencies (this might take time):
```bash
pip install ultralytics
```

Run the training script (using 20 epochs):

```python
from ultralytics import YOLO

# Update path to your dataset location
dataset_dir = '.cache/kagglehub/datasets/kushagrapandya/barcode-detection/versions/1'

model = YOLO('yolov8s.pt')
results = model.train(data=f'{dataset_dir}/data.yaml', epochs=20, imgsz=640, batch=8, name='retrain_yolov8s')
success = model.export(format='onnx', opset=11)
```

The trained ONNX model is saved to `~/runs/detect/retrain_yolov8s/weights/best.onnx`.

## Hailo Compilation

### Prerequisites

1. Download **Hailo Dataflow Compiler (DFC)** and **Hailo Model Zoo (HMZ)** from the [Developer Zone](https://hailo.ai/developer-zone/software-downloads/) (two `.whl` files)

2. Install both packages using pip (virtual environment recommended):
   ```bash
   pip install hailo_dataflow_compiler-*.whl hailo_model_zoo-*.whl
   ```

3. Download the YAML configuration from the [networks configuration directory](https://github.com/hailo-ai/hailo_model_zoo/tree/833ae6175c06dbd6c3fc8faeb23659c9efaa2dbe/hailo_model_zoo/cfg/networks): `yolov8s.yaml`

4. Set up the NMS configuration file:

    ```bash
    cd ~/lib/python3.12/site-packages/hailo_model_zoo/cfg/
    mkdir -p postprocess_config
    ```
    
    To obtain `yolov8s_nms_config.json`:
    - Locate the zip URL in the YAML file above
    - Download and extract the archive
    - Copy the JSON file to the `postprocess_config` directory

### Compile the model

This process can take several hours.

**Important notes:**

- **Hardware target**: This example targets Hailo 10H. Other platforms (Hailo 8, 8L) are also supported
- **Calibration data**: Use the validation set (`valid`) as it represents unseen data suitable for model optimization
- **GPU requirement**: If no GPU is available, set `export CUDA_VISIBLE_DEVICES=""`. This reduces optimization level to 0, which may impact accuracy and is not recommended for production

**Compilation command:**

```bash
# Optional: force CPU-only execution
export CUDA_VISIBLE_DEVICES=""  

# modify paths
hailomz compile \
    --ckpt ~/runs/detect/retrain_yolov8s/weights/best.onnx \
    --calib-path ~/.cache/kagglehub/datasets/kushagrapandya/barcode-detection/versions/1/valid \
    --yaml yolov8s.yaml \
    --classes 2 \
    --hw-arch hailo10h \
    --performance
```

Expected warnings when running without GPU:
```
[warning] Reducing optimization level to 0 (accuracy won't be optimized and compression won't be used) because there's no available GPU
[warning] Running model optimization with zero level of optimization is not recommended for production use and might lead to suboptimal accuracy results
```

## Deployment

### Understanding class mapping

**Important:** Hailo conversion adds a background class at index 0, shifting all class IDs.

**Original YOLO classes:**
- Class 0: Barcode
- Class 1: QR Code

**After Hailo conversion:**
- Class 0: unlabeled (background)
- Class 1: Barcode
- Class 2: QR Code

### Run inference

The compiled `yolov8s.hef` file is ready for deployment on the Raspberry Pi 5 AI Kit or compatible platforms.

Use `--hef-path` to specify your custom model. By default, the application uses [COCO labels](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) (80 classes). For custom models, use `--labels-json` to load your label file.

#### Download pre-trained example

```bash
hailo-download-resources --group retrain
```

This downloads the trained model from this example, including `resources/json/barcode_labels.json`.

#### Example command

```bash
python hailo_apps/python/pipeline_apps/detection/detection.py \
    --labels-json resources/json/barcode_labels.json \
    --hef-path resources/models/hailo8l/yolov8s-hailo8l-barcode.hef \
    --input resources/videos/barcode.mp4
```

#### Example output

![Example output](../images/barcode-example.png)
