"""ResNet-50 feature extraction and KMeans clustering for drawing review."""

from collections import Counter

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm


def load_images(image_list: dict[str, list[str]]) -> tuple[list[Image.Image], list[str]]:
    """Load images from a mapping of filename → [path, ...].

    Parameters
    ----------
    image_list:
        Dict mapping form identifier to a list whose first element is the image path.

    Returns
    -------
    images:
        List of loaded PIL images (RGB).
    filenames:
        Corresponding list of form identifiers.
    """
    images: list[Image.Image] = []
    filenames: list[str] = []
    for filename, paths in image_list.items():
        img = Image.open(paths[0]).convert("RGB")
        images.append(img)
        filenames.append(filename)
    return images, filenames


def extract_features_resnet(images: list[Image.Image]) -> np.ndarray:
    """Extract flattened ResNet-50 penultimate-layer features for each image.

    Parameters
    ----------
    images:
        List of PIL images.

    Returns
    -------
    np.ndarray of shape (N, 2048).
    """
    import torch
    import torchvision.transforms as transforms
    from torchvision import models

    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(base.children())[:-1])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features: list[np.ndarray] = []
    for image in tqdm(images, desc="Extracting ResNet features"):
        tensor: torch.Tensor = transform(image)  # type: ignore[assignment]
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            feature = model(tensor).squeeze().numpy()
        features.append(feature.flatten())
    return np.array(features)


def cluster_images(
    features: np.ndarray, filenames: list[str], n_clusters: int = 2
) -> tuple[dict[str, int], int]:
    """Cluster image feature vectors into *n_clusters* groups.

    Parameters
    ----------
    features:
        Feature matrix of shape (N, D).
    filenames:
        Form identifiers corresponding to each row of *features*.
    n_clusters:
        Number of KMeans clusters.

    Returns
    -------
    labels:
        Dict mapping filename → cluster label.
    majority_label:
        The cluster label that contains the most images.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    raw_labels: np.ndarray = kmeans.fit_predict(features)
    majority_label = int(Counter(raw_labels.tolist()).most_common(1)[0][0])
    return dict(zip(filenames, raw_labels.tolist(), strict=True)), majority_label


def group_images(
    image_list: dict[str, list[str]],
) -> tuple[dict[str, int], int]:
    """Load, featurise, and cluster a set of drawings.

    Parameters
    ----------
    image_list:
        Dict mapping form identifier to [image_path].

    Returns
    -------
    labels:
        Dict mapping filename → cluster label.
    majority_label:
        The dominant cluster label.
    """
    images, filenames = load_images(image_list)
    features = extract_features_resnet(images)
    return cluster_images(features, filenames)


def detailed_drawing_adjuestment(  # noqa: N802 – preserve original spelling
    reference_image_paths: list[str],
    image_candidates: list[str],
) -> list[str]:
    """Re-rank candidate drawings by Euclidean distance to reference (majority) drawings.

    Parameters
    ----------
    reference_image_paths:
        Paths to the majority-cluster (reference) drawings.
    image_candidates:
        Paths to minority-cluster drawings to be re-ranked.

    Returns
    -------
    List of candidate paths sorted by ascending average distance to the references.
    """
    refs = [Image.open(p).convert("RGB") for p in reference_image_paths]
    candidates = [Image.open(p).convert("RGB") for p in image_candidates]
    feat_candidates = extract_features_resnet(candidates)
    feat_refs = extract_features_resnet(refs)
    distances = cdist(feat_refs, feat_candidates, metric="euclidean")
    avg_distances = np.mean(distances, axis=0)
    sorted_pairs = sorted(zip(avg_distances, image_candidates), key=lambda x: x[0])
    return [path for _, path in sorted_pairs]
