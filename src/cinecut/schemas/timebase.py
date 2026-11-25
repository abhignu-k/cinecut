"""Rational frame-rate and millisecond timebase utilities."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RationalFPS(BaseModel):
    """Exact rational frame rate (e.g., 30000/1001 for 29.97 fps)."""

    num: int = Field(gt=0, description="Frame rate numerator.")
    den: int = Field(gt=0, description="Frame rate denominator.")

    def as_float(self) -> float:
        """Return the floating-point frames-per-second value."""
        return self.num / self.den

    def frame_duration_ms(self) -> float:
        """Return the duration of one frame in milliseconds."""
        return (1000.0 * self.den) / self.num

    def ms_to_frame(self, ms: int) -> int:
        """Convert integer milliseconds to the nearest frame index."""
        return round((ms * self.num) / (1000.0 * self.den))

    def frame_to_ms(self, frame: int) -> int:
        """Convert a frame index to the nearest integer milliseconds."""
        return round((frame * 1000.0 * self.den) / self.num)
