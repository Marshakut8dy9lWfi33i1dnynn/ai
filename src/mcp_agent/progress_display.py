"""
Centralized progress display configuration for MCP Agent.
Provides a shared progress display instance for consistent progress handling.
"""

from mnemo_agent.console import console
from mnemo_agent.logging.rich_progress import RichProgressDisplay

# Main progress display instance - shared across the application
progress_display = RichProgressDisplay(console)
