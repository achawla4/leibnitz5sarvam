# -*- coding: utf-8 -*-
"""
Leibnitz Forward-Compatibility & Upgradability Framework
=========================================================
Ensures effortless upgradability to Leibnitz 8.0, 10.0, or higher.

HOW TO UPGRADE WHEN LEIBNITZ 8 / 10 RELEASES:
--------------------------------------------
1. Define your new blocks (e.g., Quantum SFT, Neural Diffusion Denoiser, Beamformer).
2. Use `@leibnitz_extension(block_id, version="8.0")` to register them.
3. The Sarvam pipeline UI and API automatically discover and expose the new blocks
   without any modification to the Sarvam client or web routing layer!
"""

from typing import Callable, Dict, Any, Optional
from .adapter import registry, LeibnitzBlock

def leibnitz_extension(block_id: str, name: str, category: str = "custom", version: str = "8.0", description: str = ""):
    """Decorator to register a future Leibnitz 8 / 10 / higher block into the Sarvam workflow."""
    def decorator(fn: Callable[[Any, float, Dict[str, Any]], Dict[str, Any]]):
        block = LeibnitzBlock(
            block_id=block_id,
            name=name,
            category=category,
            version=version,
            handler=fn,
            description=description
        )
        registry.register(block)
        return fn
    return decorator

def try_load_parent_suite():
    """
    Attempts to connect to the parent SignalProcessingSuite if available in the workspace.
    Enables backward and forward linking across Leibnitz installations.
    """
    try:
        import importlib
        parent_suite = importlib.import_module("SignalProcessingSuite.blocks")
        list_blocks_fn = getattr(parent_suite, "list_blocks", None)
        get_block_fn = getattr(parent_suite, "get_block", None)
        
        if list_blocks_fn and get_block_fn:
            for b_info in list_blocks_fn():
                bid = b_info.get("id")
                if not registry.get_block(bid):
                    # Wrap parent block
                    p_block = get_block_fn(bid)
                    def make_wrapper(blk):
                        def wrapper(signal, sample_rate, params):
                            from SignalProcessingSuite.blocks import timed_run
                            res, elapsed = timed_run(blk, signal, sample_rate, params)
                            return {
                                "output_signal": res.output_signal,
                                "metrics": res.result if isinstance(res.result, dict) else {"result": res.result},
                                "elapsed_ms": elapsed
                            }
                        return wrapper
                    
                    registry.register(LeibnitzBlock(
                        block_id=bid,
                        name=f"{b_info.get('name', bid)} (Leibnitz Core)",
                        category=b_info.get("category", "core"),
                        version="5.0",
                        handler=make_wrapper(p_block),
                        description="Inherited from parent Leibnitz SignalProcessingSuite."
                    ))
            print("[Leibnitz-Sarvam] Successfully bridged with parent SignalProcessingSuite.")
    except Exception as e:
        # Standalone mode - fully functional independently
        pass

# Attempt bridging on startup
try_load_parent_suite()
