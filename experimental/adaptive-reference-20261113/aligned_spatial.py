"""Prepared fixed mapping for certified same-frame references, not a model."""
from reference_color_app.lut import apply_mapping, _strength
from spatial_render import finish_spatial, validate_rgb


class PreparedAligned:
    def __init__(self, engine, grid, diagnostics, mapping):
        self.engine, self.grid, self.diagnostics, self.mapping = engine, grid, diagnostics, mapping

    def apply(self, rgb, strength=1., chunk_rows=128):
        validate_rgb(rgb)
        strength=float(_strength(strength))
        if type(chunk_rows)!=int or chunk_rows<1:
            raise ValueError('Positive integer band height required')
        if strength==0:return rgb.copy()
        return finish_spatial(rgb,apply_mapping(rgb,self.mapping,1),strength)
