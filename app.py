import streamlit as st

from crew.research_crew import run_research


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Research Multi-Agent Team",
    page_icon="🔬",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            opacity: 0.75;
            margin-bottom: 30px;
        }

        .agent-card {
            padding: 18px;
            border-radius: 15px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 10px;
            background: rgba(128, 128, 128, 0.05);
        }

        .research-box {
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🔬 Research Multi-Agent Team</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "A CrewAI-powered research team using Groq GPT-OSS 120B"
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# AGENT TEAM
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="agent-card">
        🧠 <b>Research Planner</b><br>
        Breaks the question into research areas.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="agent-card">
        🔎 <b>Web Researcher</b><br>
        Searches the live web for evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="agent-card">
        ✅ <b>Fact Checker</b><br>
        Reviews claims and source quality.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="agent-card">
        📝 <b>Research Writer</b><br>
        Produces the final research report.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# =========================================================
# RESEARCH INPUT
# =========================================================

st.subheader("Enter Your Research Question")

research_question = st.text_area(
    "Research Question",
    placeholder=(
        "Example: What are the effects of generative AI "
        "on higher education?"
    ),
    height=150,
)


# =========================================================
# RUN RESEARCH
# =========================================================

if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
):

    if not research_question.strip():
        st.warning(
            "Please enter a research question first."
        )

    else:
        st.session_state["research_started"] = True

        progress_box = st.empty()

        try:
            progress_box.info(
                "🔄 Research team is working..."
            )

            with st.spinner(
                "Agents are researching, fact-checking and writing..."
            ):
                result = run_research(
                    research_question
                )

            progress_box.success(
                "✅ Research completed successfully."
            )

            st.divider()

            st.subheader("📄 Final Research Report")

            st.markdown(
                str(result)
            )

            st.download_button(
                label="📥 Download Report",
                data=str(result),
                file_name="research_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as exc:

            progress_box.empty()

            st.error(
                "The research team could not complete the request."
            )

            st.exception(exc)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ About")

    st.write(
        "This application uses four specialized CrewAI agents "
        "working sequentially."
    )

    st.markdown(
        """
        **Agent workflow**

        🧠 Planner  
        ↓  
        🔎 Web Researcher  
        ↓  
        ✅ Fact Checker  
        ↓  
        📝 Writer
        """
    )

    st.divider()

    st.caption(
        "LLM: Groq GPT-OSS 120B"
    )

    st.caption(
        "Framework: CrewAI"
    )

    st.caption(
        "Interface: Streamlit"
    )
