import os, sys, uuid
sys.path.insert(0, "src")

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.types import Command

from jobagent.parser import extract_text, build_profile
from jobagent.graph import build_agent

load_dotenv()

st.set_page_config(page_title="Job Agent", page_icon="💼")
st.title("💼 Job Agent")
st.caption("Upload your resume, get matched jobs, review a drafted cover letter. "
           "Nothing is sent without your approval, and you apply yourself.")
st.info("Your resume text is sent to Groq's API to generate the profile and letter. It is not saved to disk.")


@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)


@st.cache_resource
def get_agent():
    return build_agent(get_llm())


ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("profile", None)
ss.setdefault("pending", None)   # interrupt payload
ss.setdefault("result", None)    # final decision + letter
ss.setdefault("no_match", False)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to .env.")
    st.stop()

file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if file and st.button("Find jobs", type="primary"):
    ss.thread_id = str(uuid.uuid4())
    ss.pending = ss.result = None
    ss.no_match = False
    try:
        with st.spinner("Reading your resume..."):
            text = extract_text(file.getvalue())
            if len(text) < 100:
                st.error("Could not read text from this PDF. It may be a scanned image.")
                st.stop()
            ss.profile = build_profile(get_llm(), text)
        with st.spinner("Finding and scoring jobs (about a minute)..."):
            out = get_agent().invoke({"profile": ss.profile},
                                     {"configurable": {"thread_id": ss.thread_id}})
        if "__interrupt__" in out:
            ss.pending = out["__interrupt__"][0].value
        else:
            ss.no_match = True
    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}")

if ss.profile:
    with st.expander("Your extracted profile", expanded=False):
        st.json(ss.profile)

if ss.no_match:
    st.warning("No jobs scored 30 or higher. Try a resume with more skills listed.")

if ss.pending:
    p = ss.pending
    st.subheader(f'{p["role"]} @ {p["company"]}')
    st.metric("Match score", f'{p["score"]}/100')
    c1, c2 = st.columns(2)
    c1.markdown("**Matched skills**\n\n" + (", ".join(p["matched"]) or "none"))
    c2.markdown("**Missing skills**\n\n" + (", ".join(p["missing"]) or "none"))
    st.link_button("Open job posting", p["url"])

    letter = st.text_area("Draft cover letter (edit if you like)", p["letter"], height=300)
    st.caption("Check that the letter only claims things that are true.")

    a, s = st.columns(2)
    cfg = {"configurable": {"thread_id": ss.thread_id}}
    if a.button("Approve"):
        resume = "approve" if letter == p["letter"] else letter
        final = get_agent().invoke(Command(resume=resume), cfg)
        ss.result = {"decision": "approve", "letter": final["letter"], "job": p}
        ss.pending = None
        st.rerun()
    if s.button("Skip"):
        get_agent().invoke(Command(resume="skip"), cfg)
        ss.result = {"decision": "skip"}
        ss.pending = None
        st.rerun()

if ss.result:
    if ss.result["decision"] == "approve":
        st.success("Approved. Download the letter, then open the posting and apply yourself.")
        st.download_button("Download cover letter", ss.result["letter"], file_name="cover_letter.txt")
        st.link_button("Open job posting", ss.result["job"]["url"])
    else:
        st.info("Skipped. Upload again to run a new search.")