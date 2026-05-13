"""Launcher that patches deepspeed pydantic v2 incompatibility."""
import importlib, sys

# Patch deepspeed's config_utils to work with pydantic v2
import pydantic
if int(pydantic.VERSION.split('.')[0]) >= 2:
    # Monkey-patch the get_config_default to handle pydantic v2 FieldInfo
    import deepspeed.runtime.config_utils as ds_config_utils
    _orig = ds_config_utils.get_config_default
    def _patched_get_config_default(cls, field_name):
        try:
            return _orig(cls, field_name)
        except AttributeError:
            # pydantic v2: use model_fields
            field = cls.model_fields[field_name]
            if field.default is not None:
                return field.default
            return None
    ds_config_utils.get_config_default = _patched_get_config_default
