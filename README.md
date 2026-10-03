# 🔬 Research Multi-Agent Team

A modular multi-agent research application built with CrewAI, Groq and Streamlit.

## 🤖 Agent Team

The application contains four specialized agents:

1. Research Planning Specialist
2. Web Research Specialist
3. Research Fact Checker
4. Research Report Writer

## 🔄 Workflow

User Research Question

↓

Research Planning Agent

↓

Web Research Agent

↓

Fact Checking Agent

↓

Research Writer Agent

↓

Final Research Report

## 🛠️ Technologies

- Python 3.12
- CrewAI
- Groq
- GPT-OSS 120B
- Streamlit

## 🔎 Tools

The agents use specialized tools:

- Research Planning Tool
- Web Research Tool
- Source Analyzer Tool
- Groq Browser Search

## 🔐 Environment Variable

The application requires:

GROQ_API_KEY

For Streamlit Community Cloud, add the key through the application's Secrets settings.

Example:

GROQ_API_KEY = "your_api_key_here"

Do not commit API keys to GitHub.

## 🚀 Deployment

This project is designed for deployment on Streamlit Community Cloud.

1. Upload the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Select `app.py` as the main file.
5. Select Python 3.12 in Advanced Settings.
6. Add the `GROQ_API_KEY` secret.
7. Deploy the application.

## 📁 Project Structure

```text
research-multi-agent-team/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── planner_agent.py
│   ├── researcher_agent.py
│   ├── fact_checker_agent.py
│   └── writer_agent.py
│
├── tools/
│   ├── __init__.py
│   ├── research_planning_tool.py
│   ├── web_search_tool.py
│   └── source_analyzer_tool.py
│
└── crew/
    ├── __init__.py
    └── research_crew.py
