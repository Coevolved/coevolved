import logging
from typing import Optional

from coevolved.base.tracing import BaseEvent, DefaultFormatter, EventFormatter


class ConsoleSink:
    """Lightweight default sink; logs a compact, readable line.

    Args:
        formatter: Optional formatter. Uses DefaultFormatter if not provided.
    """

    def __init__(self, formatter: Optional[EventFormatter] = None) -> None:
        self.formatter = formatter or DefaultFormatter()

    def emit(self, event: BaseEvent) -> None:
        logger = logging.getLogger("base.tracing")
        summary = self.formatter.format(event)
        if summary:
            logger.info(summary)

