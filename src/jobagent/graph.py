from typing import TypedDict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from .jobs import remoteok_jobs, safe_score, draft_cover_letter


class AgentState(TypedDict):
    profile: dict
    jobs: List[dict]
    listings: List[dict]
    selected: Optional[dict]
    letter: str
    decision: str


def build_agent(llm):
    def n_find(state):
        return {"jobs": remoteok_jobs(state["profile"]["search_keywords"])}

    def n_score(state):
        p = state["profile"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            scored = [r for r in pool.map(lambda j: safe_score(llm, p, j), state["jobs"][:15]) if r]
        listings = [j for j in scored if j["is_job_listing"] and j["match_score"] >= 20]
        listings.sort(key=lambda j: j["match_score"], reverse=True)
        return {"listings": listings}

    def n_pick(state):
        return {"selected": state["listings"][0] if state["listings"] else None}

    def n_draft(state):
        return {"letter": draft_cover_letter(llm, state["profile"], state["selected"])}

    def n_approve(state):
        j = state["selected"]
        answer = interrupt({
            "role": j["role"], "company": j["company"], "score": j["match_score"],
            "matched": j["matched_skills"], "missing": j["missing_skills"],
            "url": j["url"], "letter": state["letter"],
        })
        if answer == "skip":
            return {"decision": "skip"}
        if answer == "approve":
            return {"decision": "approve"}
        return {"decision": "approve", "letter": answer}

    def route(state):
        return "draft" if state["selected"] else END

    b = StateGraph(AgentState)
    b.add_node("find", n_find)
    b.add_node("score", n_score)
    b.add_node("pick", n_pick)
    b.add_node("draft", n_draft)
    b.add_node("approve", n_approve)
    b.add_edge(START, "find")
    b.add_edge("find", "score")
    b.add_edge("score", "pick")
    b.add_conditional_edges("pick", route, {"draft": "draft", END: END})
    b.add_edge("draft", "approve")
    b.add_edge("approve", END)
    return b.compile(checkpointer=MemorySaver())