"""
assert_close() is highly configurable with strict default settings. Users are encouraged to partial() it to fit their use case. For example, if an equality check is needed, one might define an assert_equal that uses zero tolerances for every dtype by default
"""

import torch


def compare_tensors():
    """
    Compare the tensors
    """
    # tensor to tensor comparison
    expected = torch.tensor([1.0, 2.0, 3.0])
    actual = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(actual, expected, msg="The tensors are not close!")
    return True


def tests():
    """
    Assert the PyTorch tests
    """
    assert compare_tensors()
