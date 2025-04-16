from typing import Dict, Optional
from enum import Enum

class Language(Enum):
    ENGLISH = "en"
    PERSIAN = "fa"

class LanguageConfig:
    def __init__(self, default_language: Language = Language.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[Language, str]] = {
            "error_messages": {
                Language.ENGLISH: {
                    "agent_not_found": "Agent not found",
                    "task_failed": "Task failed to execute",
                    "invalid_input": "Invalid input provided",
                },
                Language.PERSIAN: {
                    "agent_not_found": "عامل یافت نشد",
                    "task_failed": "اجرای وظیفه با شکست مواجه شد",
                    "invalid_input": "ورودی نامعتبر است",
                }
            },
            "status_messages": {
                Language.ENGLISH: {
                    "task_started": "Task started",
                    "task_completed": "Task completed",
                    "task_in_progress": "Task in progress",
                },
                Language.PERSIAN: {
                    "task_started": "وظیفه شروع شد",
                    "task_completed": "وظیفه تکمیل شد",
                    "task_in_progress": "وظیفه در حال انجام است",
                }
            }
        }

    def set_language(self, language: Language) -> None:
        """Set the current language for the system."""
        self.current_language = language

    def get_translation(self, category: str, key: str, language: Optional[Language] = None) -> str:
        """Get a translation for a specific message."""
        lang = language or self.current_language
        try:
            return self.translations[category][lang][key]
        except KeyError:
            # Fallback to English if translation not found
            return self.translations[category][Language.ENGLISH][key]

    def add_translation(self, category: str, key: str, translations: Dict[Language, str]) -> None:
        """Add new translations to the system."""
        if category not in self.translations:
            self.translations[category] = {}
        
        for lang, text in translations.items():
            if lang not in self.translations[category]:
                self.translations[category][lang] = {}
            self.translations[category][lang][key] = text

# Global language configuration instance
language_config = LanguageConfig() 