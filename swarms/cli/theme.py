"""
Swarms CLI Theming
"""

from typing import Dict, Any
from rich.theme import Theme


SWARMS_COLORS = {
    # Primary Corporate Authority Colors
    "swarms_red": "#E60000",          # Corporate directive red
    "hex_red": "#FF0000",             # Command priority red
    "carbon_black": "#0D1117",        # System background
    "panel_black": "#1A1A1A",         # Control panel background
    "border_gray": "#2D2D2D",         # System boundaries
    
    # Industrial Control Accents
    "command_red": "#FF0040",         # High priority commands
    "directive_red": "#CC0000",       # System directives
    "steel_gray": "#374151",          # Infrastructure gray
    "chrome_gray": "#6B7280",         # Secondary systems
    "control_white": "#FFFFFF",       # Primary control text
    
    # System Status Colors
    "success": "#00FF41",             # System operational
    "warning": "#FFD700",             # Caution required
    "error": "#FF0000",               # System failure
    "info": "#00BFFF",                # Information display
    "critical": "#FF4444",            # Critical system alert
    
    # Hierarchical Text Colors
    "primary_text": "#FFFFFF",        # Command authority text
    "secondary_text": "#CCCCCC",      # Secondary information
    "muted_text": "#888888",          # Background information
    "highlight": "#FF0000",           # Critical highlights
    "system_text": "#00FF41",         # System status text
}

# ASCII Art of swarms Logo
SWARMS_LOGO_ASCII = """
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@          @@@@@@@@@          @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@            @@@@@@@            @@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@              @@@@@              @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@                @@@                @@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@                   @                  @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@                  @                  @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@                @@@                @@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@              @@@@@              @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@           @@@@@@@@            @@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@                 @@@                 @@@@@@@@@@@@@@@@
@@@@@@@@@@            @@@@@@@            @@@@@@@            @@@@@@@@@@
@@@@@@@@@              @@@@@              @@@@@              @@@@@@@@@
@@@@@@@@                @@@                @@@                 @@@@@@@
@@@@@@@                  @                  @                   @@@@@@
@@@@@@@                  @                  @                   @@@@@@
@@@@@@@@                @@@                @@@                 @@@@@@@
@@@@@@@@@              @@@@@              @@@@@               @@@@@@@@
@@@@@@@@@@            @@@@@@@            @@@@@@@            @@@@@@@@@@
@@@@@@@@@@@@          @@@@@@@@           @@@@@@@@          @@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@           @@@@@@@@            @@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@              @@@@@              @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@                @@@                @@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@                  @                  @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@                   @                  @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@                @@@                @@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@              @@@@@              @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@            @@@@@@@            @@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@          @@@@@@@@@          @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""

# Text
SWARMS_CORPORATE_HEADER = """
    ███████ ██     ██  █████  ██████  ███    ███ ███████ 
    ██      ██     ██ ██   ██ ██   ██ ████  ████ ██      
    ███████ ██  █  ██ ███████ ██████  ██ ████ ██ ███████ 
         ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██      ██ 
    ███████  ███ ███  ██   ██ ██   ██ ██      ██ ███████ 
    
    CORPORATE MULTI-AGENT CONTROL MATRIX
    SYSTEMATIC INTELLIGENCE MANAGEMENT PLATFORM
    """

# Hexagon
HEXAGON_CONTROL_PATTERN = """
      ████ ████ ████
    ████ ████ ████ ████
  ████ ████ ████ ████ ████
████ ████ ████ ████ ████ ████
  ████ ████ ████ ████ ████
    ████ ████ ████ ████
      ████ ████ ████
"""

# Industrial System Borders
INDUSTRIAL_BORDER = "═" * 120
SECTION_DIVIDER = "─" * 80
SUBSECTION_DIVIDER = "." * 60

# Symbols
SYMBOLS = {
    # Command Hierarchy
    "dashboard": "■",
    "agents": "▣",
    "workflows": "▦",
    "config": "▤",
    "analytics": "▧",
    "api": "▨",
    "api_keys": "▩",
    
    # Security and Control
    "shield": "▲",
    "security": "◆",
    "lock": "◼",
    "unlock": "◻",
    "access": "►",
    "restricted": "◄",
    
    # System Status Indicators
    "active": "●",
    "idle": "○",
    "error": "✖",
    "success": "✓",
    "warning": "▲",
    "info": "●",
    "critical": "◆",
    "alert": "▼",
    
    # Command Structure
    "command": "►",
    "directive": "▶",
    "execute": "▷",
    "terminate": "◻",
    "help": "?",
    "onboarding": "►",
    
    # System Infrastructure
    "gear": "▣",
    "cpu": "■",
    "memory": "▦",
    "disk": "▧",
    "network": "▨",
    "power": "▩",
    
    # Status Hierarchy
    "status_active": "●",
    "status_idle": "○",
    "status_running": "►",
    "status_error": "✖",
    "status_critical": "◆",
    "status_warning": "▲",
    
    # Navigation Control
    "arrow_right": "►",
    "arrow_left": "◄",
    "arrow_up": "▲",
    "arrow_down": "▼",
    "bullet": "■",
    "diamond": "◆",
    "square": "■",
    "circle": "●",
    "triangle": "▲",
    "cross": "✖",
    "check": "✓",
    "plus": "▲",
    "minus": "▼",
    "pipe": "│",
    "corner": "└",
    "branch": "├",
    "line": "─",
    "vertical": "│",
    "horizontal": "─",
    
    # System Operations
    "process": "▣",
    "queue": "▦",
    "cache": "▧",
    "database": "▨",
    "file": "▩",
    "folder": "▤",
    "archive": "▥",
    "backup": "▨",
    "sync": "◈",
    "transfer": "◇",
}

def get_symbols() -> Dict[str, str]:
    """Get industrial corporate control symbols"""
    return SYMBOLS

def get_theme_colors() -> Dict[str, str]:
    """Get corporate control color palette"""
    return SWARMS_COLORS

def get_rich_theme() -> Theme:
    """Get Rich theme configuration for corporate control interface"""
    theme_styles = {
        "info": f"bold {SWARMS_COLORS['info']}",
        "warning": f"bold {SWARMS_COLORS['warning']}",
        "error": f"bold {SWARMS_COLORS['error']}",
        "success": f"bold {SWARMS_COLORS['success']}",
        "critical": f"bold {SWARMS_COLORS['critical']}",
        "primary": f"bold {SWARMS_COLORS['hex_red']}",
        "secondary": SWARMS_COLORS['steel_gray'],
        "muted": SWARMS_COLORS['muted_text'],
        "highlight": f"bold {SWARMS_COLORS['highlight']}",
        "system": f"bold {SWARMS_COLORS['system_text']}",
        "command": f"bold {SWARMS_COLORS['command_red']}",
        "directive": f"bold {SWARMS_COLORS['directive_red']}",
        "control": f"bold {SWARMS_COLORS['control_white']}",
        "panel_title": f"bold {SWARMS_COLORS['hex_red']}",
        "panel_border": SWARMS_COLORS['swarms_red'],
    }
    return Theme(theme_styles)

def get_tui_theme() -> Dict[str, Any]:
    """Get TUI theme configuration for corporate control"""
    return {
        "name": "swarms_corporate_control",
        "dark": True,
        "primary": SWARMS_COLORS["hex_red"],
        "secondary": SWARMS_COLORS["steel_gray"],
        "accent": SWARMS_COLORS["command_red"],
        "background": SWARMS_COLORS["carbon_black"],
        "surface": SWARMS_COLORS["panel_black"],
        "muted": SWARMS_COLORS["muted_text"],
        "success": SWARMS_COLORS["success"],
        "warning": SWARMS_COLORS["warning"],
        "error": SWARMS_COLORS["error"],
        "info": SWARMS_COLORS["info"],
        "critical": SWARMS_COLORS["critical"],
    }

def get_corporate_panel(content: str, title: str = "", subtitle: str = "") -> str:
    """Create a maximalist corporate control panel"""
    border_char = "═"
    side_char = "║"
    corners = ["╔", "╗", "╚", "╝"]
    
    lines = content.split('\n')
    max_width = max(len(line) for line in lines) if lines else 0
    
    if title:
        title_width = len(title) + 8
        max_width = max(max_width, title_width)
    
    if subtitle:
        subtitle_width = len(subtitle) + 8
        max_width = max(max_width, subtitle_width)
    
    panel_width = max_width + 8
    
    # Top border with corporate styling
    top = f"{corners[0]}{border_char * (panel_width - 2)}{corners[1]}"
    
    # Title section with hierarchical emphasis
    title_lines = []
    if title:
        padding = (panel_width - len(title) - 8) // 2
        title_line = f"{side_char} {' ' * padding}▣ {title} ▣{' ' * (panel_width - len(title) - padding - 8)} {side_char}"
        title_lines.append(title_line)
        title_lines.append(f"{side_char}{border_char * (panel_width - 2)}{side_char}")
    
    if subtitle:
        padding = (panel_width - len(subtitle) - 4) // 2
        subtitle_line = f"{side_char} {' ' * padding}{subtitle}{' ' * (panel_width - len(subtitle) - padding - 4)} {side_char}"
        title_lines.append(subtitle_line)
        title_lines.append(f"{side_char}{border_char * (panel_width - 2)}{side_char}")
    
    # Content lines with systematic formatting
    content_lines = []
    for line in lines:
        padding = panel_width - len(line) - 4
        if line.strip().startswith(('■', '▣', '▦', '▧', '▨', '▩')):
            # Command or status line
            formatted_line = f"{side_char} ► {line}{' ' * (padding - 2)} {side_char}"
        else:
            # Regular content line
            formatted_line = f"{side_char}   {line}{' ' * (padding - 2)} {side_char}"
        content_lines.append(formatted_line)
    
    # Bottom border
    bottom = f"{corners[2]}{border_char * (panel_width - 2)}{corners[3]}"
    
    result = [top]
    result.extend(title_lines)
    result.extend(content_lines)
    result.append(bottom)
    
    return '\n'.join(result)

def get_command_header() -> str:
    """Get the corporate command header display"""
    return f"""
{INDUSTRIAL_BORDER}
{SWARMS_CORPORATE_HEADER}
{INDUSTRIAL_BORDER}
"""

def get_status_display(status: str, level: str = "info") -> str:
    """Get formatted status display with corporate hierarchy"""
    status_symbols = {
        "operational": "● OPERATIONAL",
        "warning": "▲ CAUTION REQUIRED", 
        "error": "✖ SYSTEM FAILURE",
        "critical": "◆ CRITICAL ALERT",
        "processing": "▣ PROCESSING",
        "complete": "✓ COMPLETE",
        "pending": "○ PENDING",
    }
    
    symbol = status_symbols.get(status, f"● {status.upper()}")
    return f"[SYSTEM STATUS] {symbol}"

def get_command_prompt(command: str) -> str:
    """Get formatted command prompt with corporate styling"""
    return f"[COMMAND MATRIX] ► {command.upper()}"

def get_section_header(section: str) -> str:
    """Get formatted section header with corporate hierarchy"""
    return f"""
{SECTION_DIVIDER}
[CONTROL SECTION] ▣ {section.upper()}
{SECTION_DIVIDER}
"""

def get_subsection_header(subsection: str) -> str:
    """Get formatted subsection header"""
    return f"""
{SUBSECTION_DIVIDER}
[SUBSYSTEM] ► {subsection.upper()}
{SUBSECTION_DIVIDER}
""" 
