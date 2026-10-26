from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from ultralytics import YOLO  # type: ignore[attr-defined]

# YOLOv12n detector for detailed drawings and tables, shipped with the repository
_WEIGHTS = Path(__file__).parent.parent.parent / "weights/drawing_detection.pt"


def getModel() -> "YOLO":  # noqa: N802
    """Return the local YOLO model for detailed drawing detection."""
    from ultralytics import YOLO as _YOLO  # type: ignore[attr-defined]

    return _YOLO(str(_WEIGHTS))


def crop_image(image: Image.Image, prediction: dict, outputDir: str, index: int) -> None:  # noqa: N803
    """Crop a detailed drawing or table from the given bounding box prediction.

    Parameters
    ----------
    image:
        PIL Image of the page.
    prediction:
        Bounding box dict with keys x, y, width, height, class (centre + wh format).
    outputDir:
        Directory where the cropped image will be saved.
    index:
        Numeric suffix appended to the saved filename.
    """
    x, y, width, height, image_class = (
        prediction["x"],
        prediction["y"],
        prediction["width"],
        prediction["height"],
        prediction["class"],
    )
    cropped = image.crop((x - width // 2, y - height // 2, x + width // 2, y + height // 2))
    cropped.save(Path(outputDir) / f"{image_class}{index}.png")


def crop_page(model: YOLO, pagePath: str, outputDir: str) -> Image.Image:  # noqa: N803
    """Segment a page into detailed drawings and tables.

    Runs YOLO inference on *pagePath*, crops each detected region into
    *outputDir*, saves an annotated overview image alongside the crops, and
    returns the annotated PIL image.

    Parameters
    ----------
    model:
        Loaded YOLO model (from :func:`getModel`).
    pagePath:
        Path to the page PNG.
    outputDir:
        Directory where cropped sub-images are saved.
    """
    image = Image.open(pagePath)

    results = model.predict(pagePath, conf=0.40, iou=0.30, imgsz=640, verbose=False)
    predictions = []
    boxes = results[0].boxes
    for box in (boxes if boxes is not None else []):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        predictions.append(
            {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "class": results[0].names[int(box.cls[0])],
            }
        )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    detailed_idx = 0
    table_idx = 0
    for pred in predictions:
        x, y, w, h, cls = pred["x"], pred["y"], pred["width"], pred["height"], pred["class"]
        left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        draw.text((left, top - 10), cls, fill="red")

        if cls == "detailed drawing":
            crop_image(image, pred, outputDir, detailed_idx)
            detailed_idx += 1
        else:
            crop_image(image, pred, outputDir, table_idx)
            table_idx += 1

    annotated_path = Path(outputDir).parent / "annotated.png"
    annotated.save(annotated_path)
    return annotated
