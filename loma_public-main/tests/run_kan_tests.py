#!/usr/bin/env python
import os
import sys
import unittest
from test_kan import KANTest

if __name__ == '__main__':
    # Make sure the tests can find the Loma modules
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(KANTest)
    
    # Run tests
    print("Running KAN tests...")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful()) 