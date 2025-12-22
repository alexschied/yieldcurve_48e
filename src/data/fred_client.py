import os
from fredapi import Fred

def fred_client():
    api_key = os.getenv("FREDAPI")
    if not api_key:
        raise RuntimeError("Set FREDAPI environment variable")
    return Fred(api_key=api_key)
