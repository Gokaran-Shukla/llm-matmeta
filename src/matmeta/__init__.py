"""llm-matmeta: hybrid rule-based + LLM metadata extraction for materials-science data."""

__version__ = "0.1.0"

from .extract import extract_abstract, extract_calculation  # noqa: E402,F401
from .mapping import to_jsonld  # noqa: E402,F401
