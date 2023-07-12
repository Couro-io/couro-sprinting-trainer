"""
Store fixtures for tests
"""

import pytest

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
    
