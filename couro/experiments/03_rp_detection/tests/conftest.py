"""
Store fixtures for tests
"""

import pytest
import json

@pytest.fixture
def example_annotation() -> dict:
    """
    Example annotation dictionary
    """
    return {
        'label': 'Terminal Swing',
        'xtl': '2.0',
        'ytl': '2.0',
        'xbr': '2.0',
        'ybr': '2.0',
        'width': '200',
        'height': '200',
    }
    
@pytest.fixture
def label_encoding_dict() -> dict:
    """Label encoding dictionary"""
    label_encoding_path='./../data/processed/label_dict.json'
    with open(label_encoding_path, 'r') as file:
        label_encoding_dict = json.load(file)
    return label_encoding_dict