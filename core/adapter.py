# -*- coding: utf-8 -*-
"""
Leibnitz Core Adapter Architecture
===================================
Provides forward-compatible abstraction layer connecting Leibnitz mathematical/DSP
engines with Sarvam AI's workflow.

Designed specifically so that as Leibnitz evolves (v5.0 -> v8.0 -> v10.0+), new
computational engines (e.g. quantum operators, neural diffusion DSP, tensor decomposition)
can be registered seamlessly without breaking Sarvam API contracts or UI components.
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional, Callable

# Version specification
LEIBNITZ_SPEC_VERSION = "5.0.0"
COMPATIBLE_VERSIONS = ["5.0", "6.0", "8.0", "10.0"]

class LeibnitzBlock:
    """Standardized metadata and execution contract for any Leibnitz DSP block."""
    def __init__(self, block_id: str, name: str, category: str, version: str, 
                 handler: Callable, description: str = "", params_schema: Dict[str, Any] = None):
        self.block_id = block_id
        self.name = name
        self.category = category
        self.version = version
        self.handler = handler
        self.description = description
        self.params_schema = params_schema or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.block_id,
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "description": self.description,
            "params_schema": self.params_schema
        }

class LeibnitzCoreRegistry:
    """
    Registry for signal processing blocks across Leibnitz generations.
    Enables plug-and-play upgrades for Leibnitz 8, 10, or higher.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LeibnitzCoreRegistry, cls).__new__(cls)
            cls._instance.blocks = {}
            cls._instance.version = LEIBNITZ_SPEC_VERSION
            cls._instance.active_profile = "v5_indic_speech"
        return cls._instance

    def register(self, block: LeibnitzBlock):
        self.blocks[block.block_id] = block

    def get_block(self, block_id: str) -> Optional[LeibnitzBlock]:
        return self.blocks.get(block_id)

    def list_blocks(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for b in self.blocks.values():
            if category is None or b.category == category:
                results.append(b.to_dict())
        return results

    def set_active_version(self, version: str):
        """Allows runtime version negotiation (e.g. switching to v8.0 or v10.0 when present)."""
        self.version = version

    def execute_pipeline(self, signal: np.ndarray, sample_rate: float, pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes an ordered pipeline of Leibnitz blocks.
        Preserves signal integrity and produces stage-by-stage diagnostics.
        """
        current_signal = np.asarray(signal, dtype=float).flatten()
        stage_outputs = []
        intermediates = {}

        for idx, stage in enumerate(pipeline):
            block_id = stage.get("id")
            params = stage.get("params", {})
            block = self.get_block(block_id)
            if not block:
                raise ValueError(f"Unknown Leibnitz block: '{block_id}'. Verify plugin registration.")

            stage_result = block.handler(current_signal, sample_rate, params)
            
            # If the block transforms the time-domain signal, update current_signal
            if "output_signal" in stage_result and stage_result["output_signal"] is not None:
                current_signal = np.asarray(stage_result["output_signal"], dtype=float).flatten()

            stage_outputs.append({
                "stage": idx + 1,
                "block_id": block.block_id,
                "name": block.name,
                "version": block.version,
                "metrics": stage_result.get("metrics", {}),
                "visuals": stage_result.get("visuals", {})
            })
            intermediates[block_id] = stage_result

        return {
            "processed_signal": current_signal,
            "sample_rate": sample_rate,
            "stages": stage_outputs,
            "intermediates": intermediates,
            "engine_version": self.version
        }

# Global registry singleton
registry = LeibnitzCoreRegistry()
