#!/usr/bin/env python3
"""Test that module can be reloaded without recursion errors"""

import sys
import importlib

# Test reloading the module multiple times
for i in range(3):
    print(f"\n=== Load attempt {i+1} ===")

    # Remove module from cache to force reload
    if 'model_main_single' in sys.modules:
        del sys.modules['model_main_single']

    # Import the module
    import model_main_single

    # Test that numpy randint still works
    import numpy as np
    result = np.random.randint(0, 100)
    print(f"np.random.randint(0, 100) = {result}")
    print(f"Is patched: {hasattr(np.random.randint, '_is_patched')}")

print("\n=== All reloads successful! ===")
