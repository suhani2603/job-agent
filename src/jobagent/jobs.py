import json, re, time
import requests
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage


# ---------- 1. Get jobs ----------
def remoteok_jobs(keywords):
    data = requests.get(
        "https://remoteok.com/api",
        headers={"User-Agent": "job-agent-demo"},
        timeout=30,
    ).json()[1:]
    out = []
    for d in data:
        text = f'{d.get("position","")} {" ".join(d.get("tags", []))} {(d.get("description") or "")[:300]}'.lower()
        if any(k.lower() in text for k in keywords):
            out.append({
                "title_searched": "remoteok",
                "page_title": d.get("position"),
                "url": d.get("url"),
                "snippet": (d.get("description") or "")[:1500],
            })
    return out


# ---------- 2. Score a job ----------
class JobScore(BaseModel):
    is_job_listing: bool = Field(description="True only if the page is a specific job posting")
    company: str = Field(description="Company name, or 'unknown'")
    role: str = Field(description="Job title, or 'unknown'")
    match_score: int = Field(description="0-100 fit between the candidate and this job")
    matched_skills: List[str]
    missing_skills: List[str]
    reason: str


SCORER = SystemMessage(content=(
    "You evaluate how well a candidate fits a job posting. "
    "Use ONLY the candidate profile and the job text given. Do not invent requirements. "
    "If the text is not a specific job posting, set is_job_listing to false and match_score to 0."
))


def score_job(llm, profile, job) -> JobScore:
    cand = {k: profile[k] for k in ["skills", "experience_level", "years_of_experience", "education", "projects"]}
    schema = json.dumps(JobScore.model_json_schema())
    msg = HumanMessage(content=(
        f"Return ONLY valid JSON matching this schema, no extra text.\nSchema: {schema}\n\n"
        f"CANDIDATE:\n{json.dumps(cand)}\n\n"
        f"JOB PAGE TITLE: {job['page_title']}\nURL: {job['url']}\nTEXT: {job['snippet']}"
    ))
    raw = llm.invoke([SCORER, msg]).content
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return JobScore.model_validate_json(raw)


def safe_score(llm, profile, job, tries=3):
    for attempt in range(tries):
        try:
            return {**job, **score_job(llm, profile, job).model_dump()}
        except Exception as e:
            if "RateLimit" in type(e).__name__ and attempt < tries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            print("skip:", job["url"], "|", type(e).__name__)
            return None


# ---------- 3. Draft a cover letter ----------
WRITER = SystemMessage(content=(
    "You write short, honest cover letters. "
    "Use ONLY facts from the candidate profile. Never invent experience, employers, tools or numbers. "
    "If the job asks for a skill the candidate lacks, do not claim it; emphasise related skills and willingness to learn. "
    "Keep it under 180 words, professional, no placeholders."
))


def draft_cover_letter(llm, profile, job):
    cand = {k: profile[k] for k in ["name", "summary", "skills", "education", "projects"]}
    msg = HumanMessage(content=(
        f"CANDIDATE:\n{cand}\n\n"
        f"JOB: {job['role']} at {job['company']}\n"
        f"JOB TEXT: {job['snippet']}\n"
        f"MATCHED SKILLS: {job['matched_skills']}\n"
        f"MISSING SKILLS: {job['missing_skills']}\n\n"
        "Write the cover letter."
    ))
    return llm.invoke([WRITER, msg]).content