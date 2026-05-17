# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Lamarck — a 70B model whose only job is to raise better children.

Scaffold-only at v0.0.0. See DESIGN.md for the architecture, the
safety boundaries, and the milestone roadmap. M1 (G0 inference
harness on RunPod A100) is the first thing that actually runs.
"""

__version__ = "0.0.0"

# G0 — the fixed base model the whole generational stack starts from.
G0_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
