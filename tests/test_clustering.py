"""Tests for compliance_checker.clustering.

ResNet-50 and torch are mocked so tests run without GPU or model weights.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    """Save a small RGB PNG and return its path."""
    p = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color=(128, 64, 32)).save(p)
    return p


@pytest.fixture
def image_list(tmp_path: Path) -> dict[str, list[str]]:
    """Dict with two entries, each pointing to a tiny PNG."""
    paths = []
    for i in range(2):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (10, 10), color=(i * 50, i * 80, 200)).save(p)
        paths.append(str(p))
    return {"form_A": [paths[0]], "form_B": [paths[1]]}


# ── load_images ──────────────────────────────────────────────────────────────


class TestLoadImages:
    def test_returns_images_and_filenames(self, image_list: dict[str, list[str]]) -> None:
        from compliance_checker.clustering import load_images

        images, filenames = load_images(image_list)
        assert len(images) == 2
        assert set(filenames) == {"form_A", "form_B"}

    def test_images_are_rgb(self, image_list: dict[str, list[str]]) -> None:
        from compliance_checker.clustering import load_images

        images, _ = load_images(image_list)
        for img in images:
            assert img.mode == "RGB"

    def test_empty_dict_returns_empty_lists(self) -> None:
        from compliance_checker.clustering import load_images

        images, filenames = load_images({})
        assert images == []
        assert filenames == []


# ── cluster_images ───────────────────────────────────────────────────────────


class TestClusterImages:
    def test_returns_label_dict_and_majority_label(self) -> None:
        from compliance_checker.clustering import cluster_images

        features = np.array([[1.0, 0.0], [1.1, 0.0], [0.0, 1.0]])
        filenames = ["a", "b", "c"]
        labels, majority = cluster_images(features, filenames, n_clusters=2)
        assert set(labels.keys()) == {"a", "b", "c"}
        assert majority in set(labels.values())

    def test_majority_label_is_most_common(self) -> None:
        from compliance_checker.clustering import cluster_images

        # Two points near origin, one far — origin cluster should be majority
        features = np.array([[0.0, 0.0], [0.1, 0.0], [100.0, 100.0]])
        filenames = ["a", "b", "c"]
        labels, majority = cluster_images(features, filenames, n_clusters=2)
        majority_count = sum(1 for v in labels.values() if v == majority)
        assert majority_count == 2

    def test_all_filenames_have_a_label(self) -> None:
        from compliance_checker.clustering import cluster_images

        rng = np.random.default_rng(0)
        features = rng.standard_normal((5, 4))
        filenames = [f"f{i}" for i in range(5)]
        labels, _ = cluster_images(features, filenames)
        assert set(labels.keys()) == set(filenames)

    def test_single_cluster_raises_or_degrades_gracefully(self) -> None:
        """n_clusters=1 is technically valid — all points get label 0."""
        from compliance_checker.clustering import cluster_images

        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        filenames = ["x", "y"]
        labels, majority = cluster_images(features, filenames, n_clusters=1)
        assert majority == 0
        assert all(v == 0 for v in labels.values())


# ── detailed_drawing_adjuestment ─────────────────────────────────────────────


class TestDetailedDrawingAdjuestment:
    @patch("compliance_checker.clustering.extract_features_resnet")
    def test_returns_candidates_sorted_by_distance(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        from compliance_checker.clustering import detailed_drawing_adjuestment

        # Create dummy image files
        ref_path = str(tmp_path / "ref.png")
        cand_paths = [str(tmp_path / f"cand{i}.png") for i in range(3)]
        for p in [ref_path] + cand_paths:
            Image.new("RGB", (10, 10)).save(p)

        # extract_features_resnet is called: candidates first, then refs
        mock_extract.side_effect = [
            np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),  # candidates: d=3,1,2 from origin
            np.array([[0.0, 0.0]]),                            # refs: origin
        ]

        result = detailed_drawing_adjuestment([ref_path], cand_paths)
        # Closest first: cand1 (d=1), cand2 (d=2), cand0 (d=3)
        assert result == [cand_paths[1], cand_paths[2], cand_paths[0]]

    @patch("compliance_checker.clustering.extract_features_resnet")
    def test_single_candidate_returns_list_of_one(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        from compliance_checker.clustering import detailed_drawing_adjuestment

        ref_path = str(tmp_path / "ref.png")
        cand_path = str(tmp_path / "cand.png")
        for p in [ref_path, cand_path]:
            Image.new("RGB", (10, 10)).save(p)

        mock_extract.side_effect = [
            np.array([[0.0, 0.0]]),
            np.array([[5.0, 5.0]]),
        ]

        result = detailed_drawing_adjuestment([ref_path], [cand_path])
        assert result == [cand_path]

    @patch("compliance_checker.clustering.extract_features_resnet")
    def test_multiple_refs_use_average_distance(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        from compliance_checker.clustering import detailed_drawing_adjuestment

        refs = [str(tmp_path / f"ref{i}.png") for i in range(2)]
        cands = [str(tmp_path / f"cand{i}.png") for i in range(2)]
        for p in refs + cands:
            Image.new("RGB", (10, 10)).save(p)

        # extract_features_resnet is called: candidates first, then refs
        # refs: [0,0] and [2,0]; cand0=[10,0], cand1=[1,0]
        # avg dist ref->cand0 = (10+8)/2=9, ref->cand1 = (1+1)/2=1
        mock_extract.side_effect = [
            np.array([[10.0, 0.0], [1.0, 0.0]]),   # candidates
            np.array([[0.0, 0.0], [2.0, 0.0]]),    # refs
        ]

        result = detailed_drawing_adjuestment(refs, cands)
        assert result == [cands[1], cands[0]]


# ── group_images (integration — mock ResNet) ─────────────────────────────────


class TestGroupImages:
    @patch("compliance_checker.clustering.extract_features_resnet")
    def test_returns_labels_and_majority(
        self, mock_extract: MagicMock, image_list: dict[str, list[str]]
    ) -> None:
        from compliance_checker.clustering import group_images

        mock_extract.return_value = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels, majority = group_images(image_list)

        assert set(labels.keys()) == set(image_list.keys())
        assert majority in set(labels.values())

    @patch("compliance_checker.clustering.extract_features_resnet")
    def test_majority_label_appears_most_often(
        self, mock_extract: MagicMock, tmp_path: Path
    ) -> None:
        from compliance_checker.clustering import group_images

        # 3 images: two similar, one outlier
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            Image.new("RGB", (10, 10)).save(p)
            paths.append(str(p))

        img_list = {f"form_{i}": [p] for i, p in enumerate(paths)}
        mock_extract.return_value = np.array([[0.0, 0.0], [0.1, 0.0], [100.0, 100.0]])

        labels, majority = group_images(img_list)
        majority_count = sum(1 for v in labels.values() if v == majority)
        assert majority_count >= 2
