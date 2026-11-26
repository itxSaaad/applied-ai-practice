import atexit

import config
from day5_observability.pattern1_dual_observability import (
    configure_if_enabled,
    flush_observability,
)

from .pattern1_pipeline import incident_investigation_pipeline

root_agent = incident_investigation_pipeline
_tracer_provider = configure_if_enabled()
if _tracer_provider is not None:
    atexit.register(flush_observability, _tracer_provider)
else:
    atexit.register(config.flush_traces)
