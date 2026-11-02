"""Standard unittest test runner for the project."""

import unittest
import sys
import os

# Ensure config cache is bootstrapped
import config

if __name__ == "__main__":
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
