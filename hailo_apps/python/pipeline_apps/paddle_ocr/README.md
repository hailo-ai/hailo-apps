# Paddle OCR Application

> ⚠️ **Beta:** This application is currently in beta. Features and APIs may change.

#### Run the OCR example:
```bash
hailo-ocr
```
To close the application, press `Ctrl+C`.

This example demonstrates Optical Character Recognition (OCR) using PaddleOCR models with Hailo AI acceleration. The pipeline detects text regions in video frames and recognizes the text content within those regions.

**Key Features:**
- **Text Detection**: Identifies text regions in video frames using a detection model
- **Text Recognition**: Recognizes and extracts text from detected regions
- **Real-time Processing**: Processes live video streams with tracking support
- **Multi-stage Pipeline**: Detection → Tracking → Cropping → Recognition

#### Running with Raspberry Pi Camera input:
```bash
hailo-ocr --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
hailo-ocr --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
hailo-ocr --input /dev/video<X>
```

For additional options, execute:
```bash
hailo-ocr --help
```

#### Running as Python script

For examples:
```bash
python paddle_ocr.py --input usb
```

#### App logic

The OCR pipeline operates in two stages:

1. **Text Detection Stage**: The detection model (PaddleOCR DB) identifies text regions in the frame, outputting bounding boxes around text areas.

2. **Text Recognition Stage**: Each detected text region is cropped and fed to the recognition model (PaddleOCR CRNN), which outputs the recognized text string.

The pipeline includes a tracker to maintain consistent text region IDs across frames, reducing redundant recognition operations.

#### Working in Python with the results

The basic idea is to utilize the pipeline's callback function. In simple terms, it can be thought of as a Python function that is invoked at the end of the pipeline when frame processing is complete.

This is the recommended location to implement your logic.

```python
def app_callback(element, buffer, user_data):
    roi = hailo.get_roi_from_buffer(buffer)
    text_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    for detection in text_detections:
        if detection.get_label() == "text_region":
            # Get OCR text result from classification
            ocr_objects = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)
            if len(ocr_objects) > 0:
                text_result = ocr_objects[0].get_label()
                confidence = detection.get_confidence()
                print(f"Detected text: {text_result} (confidence: {confidence:.2f})")
    return
```

The `user_app_callback_class` extends the base callback class with OCR-specific functionality:
- `add_ocr_result()`: Store detected text with confidence and bounding box
- `get_ocr_results()`: Retrieve all detected text in the current frame
- `clear_ocr_results()`: Clear results between frames

#### Pipeline Architecture

```
Source → OCR Detection → Tracker → Cropper → OCR Recognition → Callback → Display
```

- **OCR Detection**: Identifies text regions using DB (Differentiable Binarization) model
- **Tracker**: Maintains consistent IDs for text regions across frames
- **Cropper**: Extracts detected text regions for recognition
- **OCR Recognition**: Uses CRNN model to recognize text in cropped regions

#### Configuration

The OCR pipeline uses a configuration file located at `local_resources/ocr_config.json` which includes:
- Frequency dictionary for spell-checking
- Recognition parameters
- Post-processing settings

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)

