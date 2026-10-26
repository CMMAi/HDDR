"""Tests for compliance_checker.detection.

Heavy dependencies (YOLO, GPU) are mocked so tests run without model weights.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def small_image() -> Image.Image:
    """100×100 white RGB image."""
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


@pytest.fixture
def page_png(tmp_path: Path, small_image: Image.Image) -> Path:
    """Write a small PNG to a temp directory and return its path."""
    p = tmp_path / "page.png"
    small_image.save(p)
    return p


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ── crop_image ───────────────────────────────────────────────────────────────


class TestCropImage:
    def test_saves_file_with_expected_name(
        self, small_image: Image.Image, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_image

        prediction = {"x": 50, "y": 50, "width": 20, "height": 20, "class": "detailed drawing"}
        crop_image(small_image, prediction, str(output_dir), index=0)

        saved = output_dir / "detailed drawing0.png"
        assert saved.exists(), f"Expected {saved} to exist"

    def test_saves_table_with_correct_index(
        self, small_image: Image.Image, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_image

        prediction = {"x": 50, "y": 50, "width": 10, "height": 10, "class": "table"}
        crop_image(small_image, prediction, str(output_dir), index=3)

        assert (output_dir / "table3.png").exists()

    def test_cropped_dimensions_are_smaller_than_source(
        self, small_image: Image.Image, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_image

        prediction = {"x": 50, "y": 50, "width": 20, "height": 20, "class": "detailed drawing"}
        crop_image(small_image, prediction, str(output_dir), index=0)

        cropped = Image.open(output_dir / "detailed drawing0.png")
        assert cropped.width <= small_image.width
        assert cropped.height <= small_image.height

    def test_centre_plus_half_width_clamped_by_image_bounds(
        self, output_dir: Path
    ) -> None:
        """Crop box that extends to image edge should not raise."""
        from compliance_checker.detection import crop_image

        img = Image.new("RGB", (50, 50), color=(0, 0, 0))
        # Box centred at (45,45) with size 10×10 — right/bottom edges at (50,50)
        prediction = {"x": 45, "y": 45, "width": 10, "height": 10, "class": "detailed drawing"}
        crop_image(img, prediction, str(output_dir), index=0)
        assert (output_dir / "detailed drawing0.png").exists()


# ── crop_page ────────────────────────────────────────────────────────────────


class TestCropPage:
    def _make_mock_model(self, predictions: list[dict]) -> MagicMock:
        """Build a YOLO-shaped mock that returns *predictions* from .predict()."""
        box_list = []
        names = {}
        for i, pred in enumerate(predictions):
            box = MagicMock()
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            box.xyxy = [MagicMock(tolist=lambda x1=x, y1=y, w1=w, h1=h: [x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2])]
            box.cls = [i]
            box_list.append(box)
            names[i] = pred["class"]

        result = MagicMock()
        result.boxes = box_list
        result.names = names

        model = MagicMock()
        model.predict.return_value = [result]
        return model

    def test_returns_pil_image(self, page_png: Path, output_dir: Path) -> None:
        from compliance_checker.detection import crop_page

        model = self._make_mock_model([])
        result = crop_page(model, str(page_png), str(output_dir))
        assert isinstance(result, Image.Image)

    def test_saves_annotated_png_next_to_output_dir(
        self, page_png: Path, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_page

        model = self._make_mock_model([])
        crop_page(model, str(page_png), str(output_dir))
        assert (output_dir.parent / "annotated.png").exists()

    def test_crops_detailed_drawing_from_page(
        self, page_png: Path, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_page

        model = self._make_mock_model(
            [{"x": 50, "y": 50, "width": 20, "height": 20, "class": "detailed drawing"}]
        )
        crop_page(model, str(page_png), str(output_dir))
        assert (output_dir / "detailed drawing0.png").exists()

    def test_crops_table_from_page(self, page_png: Path, output_dir: Path) -> None:
        from compliance_checker.detection import crop_page

        model = self._make_mock_model(
            [{"x": 50, "y": 50, "width": 10, "height": 10, "class": "table"}]
        )
        crop_page(model, str(page_png), str(output_dir))
        assert (output_dir / "table0.png").exists()

    def test_multiple_detections_indexed_separately(
        self, page_png: Path, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_page

        model = self._make_mock_model(
            [
                {"x": 25, "y": 25, "width": 10, "height": 10, "class": "detailed drawing"},
                {"x": 75, "y": 75, "width": 10, "height": 10, "class": "detailed drawing"},
            ]
        )
        crop_page(model, str(page_png), str(output_dir))
        assert (output_dir / "detailed drawing0.png").exists()
        assert (output_dir / "detailed drawing1.png").exists()

    def test_no_detections_produces_annotated_only(
        self, page_png: Path, output_dir: Path
    ) -> None:
        from compliance_checker.detection import crop_page

        model = self._make_mock_model([])
        crop_page(model, str(page_png), str(output_dir))
        pngs = list(output_dir.iterdir())
        assert len(pngs) == 0, "No crop files expected when model returns no detections"
        assert (output_dir.parent / "annotated.png").exists()

    def test_none_boxes_produces_no_crops(self, page_png: Path, output_dir: Path) -> None:
        """Guard against YOLO returning None for boxes (no detections variant)."""
        from compliance_checker.detection import crop_page

        result = MagicMock()
        result.boxes = None
        result.names = {}
        model = MagicMock()
        model.predict.return_value = [result]

        crop_page(model, str(page_png), str(output_dir))
        assert list(output_dir.iterdir()) == []


# ── weights path ─────────────────────────────────────────────────────────────


class TestWeightsPath:
    def test_weights_path_is_absolute(self) -> None:
        from compliance_checker import detection

        assert detection._WEIGHTS.is_absolute()

    def test_weights_file_exists(self) -> None:
        from compliance_checker import detection

        assert detection._WEIGHTS.is_file()
