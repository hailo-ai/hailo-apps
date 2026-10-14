"""Manual smoke test — runs all three models on assets/Car.jpg."""
import sys
import cv2

from hailo_apps.python.standalone_apps.tunnelvision.inference import (
    VehicleDetector, PlateDetector, PlateOCR,
)


def main():
    img = cv2.imread("assets/Car.jpg")
    if img is None:
        print("ERROR: assets/Car.jpg not found", file=sys.stderr)
        sys.exit(1)

    vd = VehicleDetector()
    vehicles = vd.detect(img)
    print(f"vehicles: {len(vehicles)}")
    for v in vehicles:
        print(f"  {v.label} score={v.score:.2f} bbox={v.bbox}")

    pd = PlateDetector()
    if vehicles:
        x1, y1, x2, y2 = map(int, vehicles[0].bbox)
        crop = img[max(0, y1):y2, max(0, x1):x2]
    else:
        crop = img
    plates = pd.detect(crop)
    print(f"plates in first vehicle: {len(plates)}")

    if plates:
        ocr = PlateOCR()
        x1, y1, x2, y2 = map(int, plates[0].bbox)
        plate_crop = crop[max(0, y1):y2, max(0, x1):x2]
        result = ocr.read(plate_crop)
        print(f"OCR: {result.plate_string!r} conf={result.confidence:.2f}")


if __name__ == "__main__":
    main()
