"""Legacy import path; the implementation is owned by cashflow_integrated."""
import sys
from cashflow_integrated import recurring_detector as _implementation

sys.modules[__name__] = _implementation
