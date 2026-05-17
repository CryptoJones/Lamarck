# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Rubric runner implementations for Tier 2 categories.

Each runner takes ``(task, model_output)`` and returns a
``RubricResult`` per the protocol defined in
``lamarck.eval.tier2_engineering``. They live as separate modules
so the Tier 2 dispatcher can import only what it needs:

  * ``peft_loop``   - 6-point programmatic ladder (L6, this package)
  * ``custom_layer`` - binary unit-test rubric (L8)
  * ``llm_judge``    - 4-point LLM-judged rubric for curriculum-design
                       and diagnostics categories (L11)
  * ``json_mode``    - Tier 3 pass-rate scorer (L15)
"""
