"""Metadata schemas and conventions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import Field

from mdio.schemas.core import CamelCaseStrictModel


if TYPE_CHECKING:
    from mdio.schemas.chunk_grid import RectilinearChunkGrid
    from mdio.schemas.chunk_grid import RegularChunkGrid


class ChunkGridMetadata(CamelCaseStrictModel):
    """Definition of chunk grid."""

    chunk_grid: RegularChunkGrid | RectilinearChunkGrid | None = Field(
        default=None,
        description="Chunk grid specification for the array.",
    )


class VersionedMetadataConvention(CamelCaseStrictModel):
    """Data model for versioned metadata convention."""


class UserAttributes(CamelCaseStrictModel):
    """User defined attributes as key/value pairs."""

    attributes: dict[str, Any] | None = Field(
        default=None,
        description="User defined attributes as key/value pairs.",
    )
