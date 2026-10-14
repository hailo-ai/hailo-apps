# TunnelVision — DeGirum Model Choices

```
VEHICLE_DETECT_MODEL = "yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1"
PLATE_DETECT_MODEL   = "yolov8n_relu6_lp--640x640_quant_hailort_hailo8l_1"
OCR_MODEL            = "yolov8n_relu6_lp_ocr--256x128_quant_hailort_hailo8l_1"
```

## Vehicle detector — class handling

The chosen vehicle detector is the general COCO YOLOv8n (`yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1`),
quantized for `hailo8l`. It emits the standard 80-class COCO labels by string name (e.g. `car`,
`truck`, `bus`, `motorcycle`, `person`, `traffic light`, ...). For TunnelVision we keep only the
four vehicle classes — `car`, `truck`, `bus`, `motorcycle` — and discard everything else at the
filter step in `inference.py`.

Why not `yolov8n_relu6_car--640x640_quant_hailort_hailo8l_1`? It is a single-class car-only model
and would miss trucks/buses/motorcycles, which we need for the tunnel/wash entrance. There is no
model named `vehicle_detection*` in the `degirum/models_hailort` zoo on `hailo8l`, so the COCO
detector is the best fit per the preference order in the plan.

## Discovery notes

- Inference host: `@local` (Hailo-8L on this RPi).
- Cloud zoo: `degirum/models_hailort`. The zoo listing requires a cloud token — `DEGIRUM_CLOUD_TOKEN`
  is read from `/usr/local/hailo/resources/.env` and passed to `dg.connect(...)` and
  `dg.load_model(...)`. Local inference also requires the token to be installed on the system via
  `degirum token install <token>` (one-time setup; written to
  `~/.local/share/DeGirum/pysdk_cloud_token.json`). Both done.
- Plate detector and OCR are present locally under `models/` (used by `carwash_lpr`) and also visible
  in the cloud zoo listing.
- Smoke test on `assets/Car.jpg` with the chosen vehicle detector returned `car 0.5` (1 detection),
  confirming the model loads and runs end-to-end on the local Hailo device.
