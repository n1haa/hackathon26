"""Legacy import path; the implementation is owned by cashflow_integrated."""
import sys
from cashflow_integrated import risk_engine as _implementation

sys.modules[__name__] = _implementation
