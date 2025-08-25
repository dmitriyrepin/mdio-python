"""Conversion from to MDIO various other formats."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import xarray as xr
from psutil import cpu_count
from tqdm.dask import TqdmCallback

from json import dumps as json_dumps 
from rich import print as rich_print
from rich.console import Console
from rich.table import Table

from mdio.segy.blocked_io import to_segy
from mdio.segy.creation import concat_files
from mdio.segy.creation import mdio_spec_to_segy
from mdio.segy.utilities import segy_export_rechunker

try:
    import distributed
except ImportError:
    distributed = None

if TYPE_CHECKING:
    from segy.schema import SegySpec

    from mdio.core.storage_location import StorageLocation

default_cpus = cpu_count(logical=True)
NUM_CPUS = int(os.getenv("MDIO__EXPORT__CPU_COUNT", default_cpus))


def _get_dask_array(mdio_xr: xr.Dataset, var_name: str, chunks: tuple[int, ...] = None) -> da.Array:
    """Workaround if the MDIO Xarray dataset returns numpy arrays instead of Dask arrays"""
    xr_var = mdio_xr[var_name]
    # xr_var.chunks:
    # Tuple of block lengths for this dataarray’s data, in order of dimensions,
    # or None if the underlying data is not a dask array.
    if xr_var.chunks is not None:
        return xr_var.data.rechunk(chunks)
    # For some reason, a NumPy in-memory array was returned
    # HACK: Convert NumPy array to a chunked Dask array
    return da.from_array(xr_var.data, chunks=chunks)


def mdio_to_segy(  # noqa: PLR0912, PLR0913, PLR0915
    segy_spec: SegySpec,
    input_location: StorageLocation,
    output_location: StorageLocation,
    endian: str = "big",
    new_chunks: tuple[int, ...] = None,
    selection_mask: np.ndarray = None,
    client: distributed.Client = None,
) -> None:
    """Convert MDIO file to SEG-Y format.

    We export N-D seismic data to the flattened SEG-Y format used in data transmission.

    The input headers are preserved as is, and will be transferred to the output file.

    Input MDIO can be local or cloud based. However, the output SEG-Y will be generated locally.

    A `selection_mask` can be provided (same shape as spatial grid) to export a subset.

    Args:
        segy_spec: The SEG-Y specification to use for the conversion.
        input_location: Store or URL (and cloud options) for MDIO file.
        output_location: Path to the output SEG-Y file.
        endian: Endianness of the input SEG-Y. Rev.2 allows little endian. Default is 'big'.
        new_chunks: Set manual chunksize. For development purposes only.
        selection_mask: Array that lists the subset of traces
        client: Dask client. If `None` we will use local threaded scheduler. If `auto` is used we
            will create multiple processes (with 8 threads each).

    Raises:
        ImportError: if distributed package isn't installed but requested.
        ValueError: if cut mask is empty, i.e. no traces will be written.

    Examples:
        To export an existing local MDIO file to SEG-Y we use the code snippet below. This will
        export the full MDIO (without padding) to SEG-Y format using IBM floats and big-endian
        byte order.

        >>> from mdio import mdio_to_segy
        >>>
        >>>
        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ... )

        If we want to export this as an IEEE big-endian, using a selection mask, we would run:

        >>> mdio_to_segy(
        ...     mdio_path_or_buffer="prefix2/file.mdio",
        ...     output_segy_path="prefix/file.segy",
        ...     selection_mask=boolean_mask,
        ... )

    """
    output_segy_path = Path(output_location.uri)

    mdio_xr = xr.open_dataset(input_location.uri, engine="zarr", mask_and_scale=False)

    trace_variable_name = mdio_xr.attrs["attributes"]["traceVariableName"]
    amplitude = mdio_xr[trace_variable_name]
    chunks: tuple[int, ...] = amplitude.encoding.get("chunks")
    shape: tuple[int, ...] = amplitude.shape
    dtype = amplitude.dtype
    if new_chunks is None:
        new_chunks = segy_export_rechunker(chunks, shape, dtype)
    mdio_xr.close()

    creation_args = [segy_spec, input_location, output_location, endian]

    if client is not None:
        if distributed is not None:
            # This is in case we work with big data
            feature = client.submit(mdio_spec_to_segy, *creation_args)
            mdio_xr, segy_factory = feature.result()
        else:
            msg = "Distributed client was provided, but `distributed` is not installed"
            raise ImportError(msg)
    else:
        mdio_xr, segy_factory = mdio_spec_to_segy(*creation_args)

    # Using XArray.DataArray.values should trigger compute and load the whole array into memory.
    live_mask = mdio_xr["trace_mask"].values
    # live_mask = mdio.live_mask.compute()

    if selection_mask is not None:
        live_mask = live_mask & selection_mask

    # This handles the case if we are skipping a whole block.
    if live_mask.sum() == 0:
        msg = "No traces will be written out. Live mask is empty."
        raise ValueError(msg)

    # Find rough dim limits, so we don't unnecessarily hit disk / cloud store.
    # Typically, gets triggered when there is a selection mask
    dim_slices = ()
    live_nonzeros = live_mask.nonzero()
    for dim_nonzeros in live_nonzeros:
        start = np.min(dim_nonzeros)
        stop = np.max(dim_nonzeros) + 1
        dim_slices += (slice(start, stop),)

    # Lazily pull the data with limits now, and limit mask so its the same shape.
    # Workaround: currently the MDIO Xarray dataset returns numpy arrays instead of Dask arrays
    # TODO (Dmitriy Repin): Revisit after the eager memory allocation is fixed
    # https://github.com/TGSAI/mdio-python/issues/608
    live_mask = _get_dask_array(mdio_xr, "trace_mask", new_chunks[:-1])[dim_slices]
    headers = _get_dask_array(mdio_xr, "headers", new_chunks[:-1])[dim_slices]
    samples = _get_dask_array(mdio_xr, "amplitude", new_chunks)[dim_slices]

    if selection_mask is not None:
        selection_mask = selection_mask[dim_slices]
        live_mask = live_mask & selection_mask

    # tmp file root
    out_dir = output_segy_path.parent
    tmp_dir = TemporaryDirectory(dir=out_dir)

    with tmp_dir:
        with TqdmCallback(desc="Unwrapping MDIO Blocks"):
            block_records = to_segy(
                samples=samples,
                headers=headers,
                live_mask=live_mask,
                segy_factory=segy_factory,
                file_root=tmp_dir.name,
            )

            if client is not None:
                block_records = block_records.compute()
            else:
                block_records = block_records.compute(num_workers=NUM_CPUS)

        ordered_files = [rec.path for rec in block_records.ravel() if rec != 0]
        ordered_files = [output_segy_path] + ordered_files

        if client is not None:
            _ = client.submit(concat_files, paths=ordered_files).result()
        else:
            concat_files(paths=ordered_files, progress=True)

def mdio_to_info_cli(mdio_path: str, storage_options: str, output_format: str="pretty") -> dict[str, Any]:
    sl = StorageLocation(uri=mdio_path, options=storage_options)
    return mdio_to_info(input_location=sl, output_format=output_format)

def mdio_to_info(input_location: StorageLocation, output_format: str=None) -> dict[str, Any]:
    """Returns information on a MDIO v1 dataset.

    Optionally, prints the information in JSON or Table formats.

    Args:
        input_location (StorageLocation): The input location of the MDIO dataset.
        output_format (str, optional): The output format for the information. Defaults to None.
            If output_format is set to 'pretty' then a human-readable table is printed.
            If output_format is set to 'json' then a JSON is printed.
            Otherwise, no output is produced.

    Returns:
        dict[str, Any]: A dictionary containing information about the MDIO dataset.
    """

    mdio_xr = xr.open_dataset(input_location.uri, engine="zarr", mask_and_scale=False)

    grid_dict = _parse_grid(mdio_xr)

    t_var_name = mdio_xr.attrs["attributes"]["traceVariableName"]
    stats = json.loads(mdio_xr[t_var_name].attrs["statsV1"])

    mdio_info = {
        "path": input_location.uri,
        "stats": stats,
        "grid": grid_dict
    }

    if output_format is not None:
        if output_format.lower() == "pretty":
            _pretty_print(mdio_info, t_var_name)
        elif output_format.lower() == "json":
            print(json_dumps(mdio_info, indent=2))

    return mdio_info

def _parse_grid(mdio_xr: xr.Dataset) -> dict[str, dict[str, int | str]]:
    """Extract grid information per dimension."""
    coords_names = list(mdio_xr.coords.keys())
    grid_dict = {}
    dimensions = []
    for dim_name in mdio_xr.dims:
        dim_var = mdio_xr[dim_name]
        t = str(dim_var.dtype)
        min_ = str(dim_var.values[0])
        max_ = str(dim_var.values[-1])
        size = str(dim_var.shape[0])
        dimensions.append({"name": dim_name, "dtype": t, "min": min_, "max": max_, "size": size})
        coords_names.remove(dim_name)
    grid_dict["dimensions"] = dimensions

    coordinates = []
    for coord_name in coords_names:
        coord_var = mdio_xr[coord_name]
        t = str(coord_var.dtype)
        d = list(coord_var.dims)
        s = list(coord_var.shape)
        ch = list(coord_var.encoding.get("chunks"))
        coordinates.append({"name": coord_var.name, "dtype": t, "dims": d, "shape": s, "chunks": ch})
    grid_dict["coordinates"] = coordinates

    variables = []
    for var in mdio_xr.data_vars.values():
        if hasattr(var.dtype, "fields") and var.dtype.fields is not None:
            t = f"Structured[{len(var.dtype.fields)}]"
        else:
            t = str(var.dtype)
        d = list(var.dims)
        c = list(set(var.coords.keys()) - set(var.dims))
        s = list(var.shape)
        ch = list(var.encoding.get("chunks"))
        variables.append({"name": var.name, "dtype": t, "dims": d, "coords": c, "shape": s, "chunks": ch})
    grid_dict["variables"] = variables

    return grid_dict

def _pretty_print(mdio_info: dict[str, Any], traceVariableName: str) -> None:
    """Print pretty MDIO Info table to console."""

    console = Console() 

    grid_table = Table(title="Dimensions", show_edge=True, expand=True)
    grid_table.add_column("Name", justify="right", style="cyan", no_wrap=True)
    grid_table.add_column("Type", justify="left", style="magenta")
    grid_table.add_column("Min", justify="left", style="magenta")
    grid_table.add_column("Max", justify="left", style="magenta")
    grid_table.add_column("Size", justify="left", style="green")

    for dim_dict in mdio_info["grid"]["dimensions"]:
        name, type_, min_, max_, size = dim_dict.values()
        grid_table.add_row(name, type_, min_, max_, size)

    coord_table = Table(title="Coordinates", show_edge=True, expand=True)
    coord_table.add_column("Name", justify="right", style="cyan", no_wrap=True)
    coord_table.add_column("Type", justify="left", style="magenta")
    coord_table.add_column("Dimensions", justify="left", style="magenta")
    coord_table.add_column("Size", justify="left", style="magenta")
    coord_table.add_column("Chunks", justify="left", style="magenta")

    for coord_dict in mdio_info["grid"]["coordinates"]:
        name, type_, dims, shape, chunks = coord_dict.values()
        coord_table.add_row(name, type_, "\n".join(dims), "\n".join(map(str, shape)), 
                            "\n".join(map(str, chunks)), end_section=True)

    var_table = Table(title="Variables", show_edge=True, expand=True)
    var_table.add_column("Name", justify="right", style="cyan", no_wrap=True)
    var_table.add_column("Type", justify="left", style="magenta")
    var_table.add_column("Dimensions", justify="left", style="magenta")
    var_table.add_column("Size", justify="left", style="magenta")
    var_table.add_column("Chunks", justify="left", style="magenta")
    var_table.add_column("Coordinates", justify="left", style="magenta")

    for var_dict in mdio_info["grid"]["variables"]:
        name, type_, dims, coords, shape, chunks = var_dict.values()
        var_table.add_row(name, type_, "\n".join(dims), "\n".join(map(str, shape)), 
                          "\n".join(map(str, chunks)), " ".join(coords), end_section=True)

    stat_table = Table(title=f"'{traceVariableName}' statistics", show_edge=False)
    stat_table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
    stat_table.add_column("Value", justify="left", style="magenta")

    for stat, value in mdio_info["stats"].items():
        if isinstance(value, float):
            stat_table.add_row(stat, f"{value:.4f}")
        # Ignore histogram for now

    master_table = Table(title=f"File: {mdio_info['path']}", expand=True)
    master_table.add_row(grid_table)
    master_table.add_row(coord_table)
    master_table.add_row(var_table)
    # master_table.add_column("Statistics", justify="center")
    master_table.add_row(stat_table)

    console.print(master_table)
