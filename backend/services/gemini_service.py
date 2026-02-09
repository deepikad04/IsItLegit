"""
Backward-compatible re-export from the gemini/ package.

The original monolithic gemini_service.py (~4000 lines) has been split into:
  - gemini/schemas.py   — Pydantic output schemas (~350 lines)
  - gemini/helpers.py   — Module-level helpers, caches, constants (~135 lines)
  - gemini/service.py   — GeminiService class (~3600 lines)

All imports that previously used `from services.gemini_service import X`
continue to work unchanged via this re-export.
"""

# Re-export everything that consumers previously imported from this module
from services.gemini.service import GeminiService  # noqa: F401
from services.gemini.helpers import (  # noqa: F401
    THINKING_LEVELS,
    _cache,
    _context_cache_store,
    _decision_trace,
    _scenario_summary,
    _cache_key,
)
from services.gemini.schemas import (  # noqa: F401
    _CredibilityCheckOutput,
    _URLScenarioOutput,
)
