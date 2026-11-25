import atexit

import config

from .pattern1_pipeline import incident_investigation_pipeline

root_agent = incident_investigation_pipeline

atexit.register(config.flush_traces)
