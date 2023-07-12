"""
Unit tests for data processing steps
"""

import sys
import pytest

sys.path.append('./couro/experiments/03_rp_detection/preprocessing')
from ..preprocessing import get_files_with_annotations, split_train_test_files, calc_x_y_center, get_key_from_value, extract_label

def test_calc_x_y_center(example_annotation):
    """
    Test the calculation of the center of a bounding box.
    """
    x_center, y_center = calc_x_y_center(
        xtl=example_annotation['xtl'], 
        ytl=example_annotation['ytl'], 
        xbr=example_annotation['xbr'], 
        ybr=example_annotation['ybr']
    )
    assert x_center == 2.0
    assert y_center == 2.0

def test_get_key_from_value():
    """
    """
    pass

def test_extract_label():
    """
    """
    pass

def test_get_files_with_annotations():
    """
    """
    pass

def get_files_with_annotations():
    """
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__])