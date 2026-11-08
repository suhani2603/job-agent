import json, re
from typing import List, Optional
import pymupdf
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage


class Profile(BaseModel):
    name: str = Field(description="Candidate's full name")
    summary: str = Field(description="2-3 sentence summary using only facts from the resume")
    skills: List[str] = Field(description="Skills and tools listed or clearly used")
    experience_level: str = Field(description="One of: student, entry-level, mid-level, senior")
    years_of_experience: float = Field(description="Years of professional experience, 0 if none")
    education: List[str]
    projects: List[str]
    target_job_titles: List[str] = Field(description="5 realistic job titles for this candidate")
    search_keywords: List[str] = Field(description="8-10 keywords for searching jobs")
    location: Optional[str] = None


SYSTEM = SystemMessage(content=(
    "You extract structured data from resumes. "
    "Use ONLY information present in the resume. Never invent skills, employers or dates. "
    "Base target_job_titles and search_keywords on the skills and projects shown."
))


def extract_text(pdf_bytes: bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc).strip()


def build_profile(llm, resume_text: str) -> dict:
    schema = json.dumps(Profile.model_json_schema())
    msg = HumanMessage(content=(
        f"Return ONLY valid JSON matching this schema, no extra text.\n"
        f"Schema: {schema}\n\nResume:\n{resume_text[:8000]}"
    ))
    raw = llm.invoke([SYSTEM, msg]).content
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return Profile.model_validate_json(raw).model_dump()