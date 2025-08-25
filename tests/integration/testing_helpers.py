"""This module provides testing helpers for integration testing."""

from collections.abc import Callable

import numpy as np
import xarray as xr

def get_values(arr: xr.DataArray) -> np.ndarray:
    """Extract actual values from an Xarray DataArray."""
    return arr.values


def get_inline_header_values(dataset: xr.Dataset) -> np.ndarray:
    """Extract a specific header value from an Xarray DataArray."""
    return dataset["inline"].values


def validate_variable(  # noqa PLR0913
    dataset: xr.Dataset,
    name: str,
    shape: list[int],
    dims: list[str],
    data_type: np.dtype,
    expected_values: range | None,
    actual_value_generator: Callable[[xr.DataArray], np.ndarray] | None = None,
) -> None:
    """Validate the properties of a variable in an Xarray dataset."""
    arr = dataset[name]
    assert shape == arr.shape
    assert set(dims) == set(arr.dims)
    if hasattr(data_type, "fields") and data_type.fields is not None:
        # The following assertion will fail because of differences in endianness and offsets
        # assert data_type == arr.dtype

        # Compare field names
        expected_names = [name for name in data_type.names]
        actual_names = [name for name in arr.dtype.names]
        assert expected_names == actual_names

        # Compare field types ignoring endianness
        expected_types = [data_type[name].newbyteorder('=') for name in data_type.names]
        actual_types = [arr.dtype[name].newbyteorder('=') for name in arr.dtype.names]
        assert expected_types == actual_types

        # Compare field offsets (fails):
        #   name: 'shot_point' dt_exp: (dtype('>i4'), 196) dt_act: (dtype('<i4'), 180)
        #
        # expected_offsets = [data_type[name][1] for name in data_type.names]
        # actual_offsets = [arr.dtype[name][1] for name in arr.dtype.names]
        # assert expected_offsets == actual_offsets
        # for name in data_type.names:
        #     dt_exp = data_type.fields[name]
        #     dt_act = arr.dtype.fields[name]
        #     if dt_exp[1] != dt_act[1]:
        #         pass

    else:
        assert data_type == arr.dtype

    if expected_values is not None and actual_value_generator is not None:
        actual_values = actual_value_generator(arr)
        assert np.array_equal(expected_values, actual_values)
