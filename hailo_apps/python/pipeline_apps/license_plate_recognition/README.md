## License Plate Recognition (Python)

This app runs a multi-stage GStreamer pipeline for license plate recognition:

- Vehicle detection + tracking
- License plate detection on vehicle crops
- OCR on cropped license plates

### Run

From the repo root:

- `python3 -m hailo_apps.python.pipeline_apps.license_plate_recognition.license_plate_recognition --help`

### Pipeline variants

Use `--pipeline` to pick a pipeline graph:

- `simple`: full LPR (vehicle → plate → OCR) using helper pipeline builders
- `complex`: full LPR with an explicit GStreamer graph (tees/queues/aggregators)
- `optimized`: full LPR with queue/batch optimizations and parallel display/processing branches
- `vehicle_and_lp`: vehicle detection + license-plate detection (no tracking, no OCR)
- `vehicle_only`: vehicle detection only
- `lp_only`: license-plate detection only (full-frame)
- `ocr_only`: OCR only (results are printed by the Python callback)

### JSON configs

The app accepts `--vehicle-json` / `--plate-json` as either:

- A filesystem path (absolute or relative), or
- A filename under `resources/json/`

The default configs are bundled under `hailo_apps/python/pipeline_apps/license_plate_recognition/configs/`.

### Custom `.so` paths

The app accepts `--yolo-postprocess-so`, `--croppers-so`, `--overlay-so`, and `--ocrsink-so` as either:

- A filesystem path (absolute or relative), or
- A filename under `resources/so/`
