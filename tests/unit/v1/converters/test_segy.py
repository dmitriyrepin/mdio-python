"""Unit tests for the segy converter module."""

from segy.schema import SegySpec
from segy.schema import HeaderField
from segy.standards import get_segy_standard

from tests.integration.testing_data import custom_teapot_dome_segy_spec

def field(segy_spec: SegySpec, field_name: str) -> HeaderField:
    return next((f for f in segy_spec.trace.header.fields if f.name == field_name), None)

def validate_custom_fields(custom: SegySpec) -> None:
    # NOTE: Pydentic models utilize "value comparison"
    assert field(custom, "inline") == HeaderField(name="inline", byte=17, format="int32")
    assert field(custom, "crossline") == HeaderField(name="crossline", byte=13, format="int32")
    assert field(custom, "cdp_x") == HeaderField(name="cdp_x", byte=81, format="int32")
    assert field(custom, "cdp_y") == HeaderField(name="cdp_y", byte=85, format="int32")

def test_customize_segy_specs() -> None:

    custom_spec = custom_teapot_dome_segy_spec(keep_unaltered=False)
    # SEG-Y spec has only the overwritten header fields
    assert len(custom_spec.trace.header.names) == 4
    validate_custom_fields(custom_spec)


    custom_spec = custom_teapot_dome_segy_spec(keep_unaltered=True)
    # Customized SEG-Y spec has the overwritten header fields plus unaltered spec v1.0 fields
    spec_v1 = get_segy_standard(1.0)
    # The number of fields is reduced by the number of the overwritten fields
    # The original byte locations of the fields being overwritten is not present in the custom spec
    assert len(custom_spec.trace.header.names) == len(spec_v1.trace.header.names) - 4

    # Original byte locations of the fields being overwritten
    affected_bytes = [field(spec_v1, name).byte for name in ("inline", "crossline", "cdp_x", "cdp_y")]
    # New byte locations of the fields being overwritten
    affected_bytes += [17, 13, 81, 85]
    for f in spec_v1.trace.header.fields:
        # NOTE: Pydentic models utilize "value comparison"
        if f.byte not in affected_bytes:
            assert field(custom_spec, f.name) == f
        else:
            validate_custom_fields(custom_spec)
