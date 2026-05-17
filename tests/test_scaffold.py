# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Scaffold-level sanity checks.

These exist so `pytest` is green from day one and to lock in the
design-level invariants that the rest of the project will depend
on. Real tests start landing in M1.
"""

from __future__ import annotations

import lamarck


def test_version_present():
    assert isinstance(lamarck.__version__, str)
    assert lamarck.__version__.count(".") == 2


def test_g0_is_deepseek_r1_distill_llama_70b():
    """The base-model choice is part of the design — not a config
    knob. If a future change wants to swap G0, that's a DESIGN.md
    edit + a separate branch + a documented rationale, not a quiet
    constant change.
    """
    assert lamarck.G0_MODEL_ID == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"


def test_generation_cap_is_three():
    """The generation cap is a safety boundary, not a magic number.
    Raising it requires DESIGN.md updates first.
    """
    assert lamarck.MAX_GENERATIONS == 3
