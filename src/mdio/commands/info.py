"""MDIO Dataset information command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from click import STRING
from click import Choice
from click import argument
from click import command
from click import option
from click_params import JSON

from mdio.converters.mdio import mdio_to_info_cli

if TYPE_CHECKING:
    from mdio import MDIOReader
    from mdio.core import Grid


@command(name="info")
@argument("mdio-path", type=STRING)
@option(
    "-storage",
    "--storage-options",
    required=False,
    help="Storage options for SEG-Y input file.",
    type=JSON,
)
@option(
    "-format",
    "--output-format",
    required=False,
    default="pretty",
    help="Output format. Pretty console or JSON.",
    type=Choice(["pretty", "json"]),
    show_default=True,
    show_choices=True,
)
def info(mdio_path: str, storage_options: dict[str, Any],  output_format: str) -> None:
    """Provide information on a MDIO dataset.

    By default, this returns human-readable information about the grid and stats for the dataset.
    If output-format is set to 'json' then a JSON is returned to facilitate parsing.
    """
    # Lazy import to reduce CLI startup time
    from mdio import MDIOReader  # noqa: PLC0415

    mdio_to_info_cli(mdio_path=mdio_path, storage_options=storage_options, output_format=output_format)

cli = info
