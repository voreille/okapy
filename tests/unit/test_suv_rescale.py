"""Unit tests for stored-value rescaling.

The SUV computation manual (v3.0.0), "Rescale slope and intercept", requires
both attributes to be present but only asks for a *warning* when the slope is
non-positive or the intercept is non-zero: a zero slope legitimately occurs in
marginal slices, so aborting the whole series would be worse than flagging it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from okapy.dicom.conversion.suv_common import (
    SUVComputationError,
    apply_rescale,
    get_required_float,
)


def test_rescale_applies_slope_and_intercept():
    values = apply_rescale(
        np.array([[0, 1], [2, 3]]),
        slope=0.5,
        intercept=0.0,
        emitted_warnings=set(),
    )

    np.testing.assert_allclose(values, [[0.0, 0.5], [1.0, 1.5]])
    assert values.dtype == np.float64


def test_non_zero_intercept_warns_but_still_converts(caplog):
    with caplog.at_level(logging.WARNING):
        values = apply_rescale(
            np.array([[1, 2]]),
            slope=1.0,
            intercept=-3.5,
            emitted_warnings=set(),
        )

    np.testing.assert_allclose(values, [[-2.5, -1.5]])

    message = caplog.text
    assert "RescaleIntercept is -3.5" in message
    assert "interpreted with caution" in message


def test_non_positive_slope_warns_but_still_converts(caplog):
    with caplog.at_level(logging.WARNING):
        values = apply_rescale(
            np.array([[1, 2]]),
            slope=0.0,
            intercept=0.0,
            emitted_warnings=set(),
        )

    np.testing.assert_allclose(values, [[0.0, 0.0]])

    message = caplog.text
    assert "RescaleSlope is 0" in message
    assert "not trustworthy" in message


def test_warnings_are_emitted_once_per_image(caplog):
    emitted: set[str] = set()

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            apply_rescale(
                np.array([[1]]),
                slope=-1.0,
                intercept=2.0,
                emitted_warnings=emitted,
            )

    # One slope warning and one intercept warning for the whole image, not per
    # slice.
    assert len(caplog.records) == 2


def test_absent_rescale_attribute_is_an_error():
    """"Must be present and not empty" is still a hard requirement."""

    class _Slice:
        pass

    with pytest.raises(SUVComputationError, match="RescaleSlope"):
        get_required_float(_Slice(), "RescaleSlope")
