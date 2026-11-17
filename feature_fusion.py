"""Feature fusion module."""
import numpy as np


def fuse_features(raw_features, statistical_features, deep_features):
    """
    Concatenate raw, statistical, and deep feature arrays.

    Note: the report's main.py passes X for both raw_features and
    statistical_features. A separate statistical-feature extraction
    implementation is not provided in the report.
    """
    fused = np.concatenate(
        (raw_features, statistical_features, deep_features),
        axis=1
    )
    return fused
