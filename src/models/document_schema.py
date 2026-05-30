from pydantic import BaseModel, Field
from typing import Optional


class InvestmentDocument(BaseModel):
    content: str = Field(..., min_length=50)
    source: Optional[str]
    doc_type: Optional[str]