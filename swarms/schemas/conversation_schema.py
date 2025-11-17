from typing import Optional
from pydantic import BaseModel, Field


class ConversationSchema(BaseModel):
    time_enabled: Optional[bool] = Field(default=False)
    message_id_on: Optional[bool] = Field(default=True)
    autosave: Optional[bool] = Field(default=False)
    count_tokens: Optional[bool] = Field(default=False)
