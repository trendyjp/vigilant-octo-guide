__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
# Corrected import path for the tool
from langchain_community.tools import DuckDuckGoSearchRun
# Required for ChatGoogleGenerativeAI
import google.generativeai as genai

# --- Streamlit UI Configuration ---
# [Rest of your Streamlit UI setup remains the same]
st.set_page_config(page_title="CrewAI Email Generator (Gemini)", layout="wide")
st.title("🚀 CrewAI Cold Email Generator (using Google Gemini)")
st.markdown("""
Generate two variations of a cold email promoting a video editing solution using a team of AI agents powered by Google Gemini.
Enter your Google API Key and click 'Generate Emails'.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google API Key", type="password", key="google_api_key",
                                   help="Get yours from Google AI Studio: https://aistudio.google.com/app/apikey")
    st.subheader("Task Customization")
    default_task_desc = '''1. Generate two distinct variations of a cold email promoting a video editing solution.
2. Evaluate the written emails for their effectiveness and engagement.
3. Scrutinize the emails for grammatical correctness and clarity.
4. Adjust the emails to align with best practices for cold outreach. Consider the feedback
provided to the marketing_strategist.
5. Revise the emails based on all feedback, creating two final versions.'''
    task_description = st.text_area("Email Task Description", value=default_task_desc, height=300, key="task_desc")
    selected_model = "gemini-pro"
    temperature = 0.5
    generate_button = st.button("Generate Emails", key="generate")

# --- Main App Logic ---
if generate_button:
    if not google_api_key:
        st.error("🚨 Please enter your Google API Key in the sidebar.")
    elif not task_description:
        st.error("🚨 Please enter a task description in the sidebar.")
    else:
        try:
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                verbose=True,
                temperature=temperature,
                google_api_key=google_api_key
            )

            # Initialize Tools (using the correct import)
            tool_search = DuckDuckGoSearchRun() # Instance creation is fine

            # Define Agents
            email_author = Agent(
                role='Professional Email Author',
                goal='Craft concise and engaging emails based on the task description',
                backstory='Experienced in writing impactful marketing emails using current best practices.',
                verbose=True,
                allow_delegation=False,
                llm=llm,
                tools=[tool_search] # Pass the instance here
            )
            marketing_strategist = Agent(
                role='Marketing Strategist',
                goal='Lead the team in creating effective cold emails based on the provided task description',
                backstory='A seasoned Chief Marketing Officer with a keen eye for standout marketing content and strategy.',
                verbose=True,
                allow_delegation=True,
                llm=llm
                # Note: Marketing strategist doesn't necessarily need the search tool directly if delegating tasks requiring it
            )
            content_specialist = Agent(
                role='Content Specialist',
                goal='Critique and refine email content for clarity, grammar, and persuasiveness',
                backstory='A professional copywriter with a wealth of experience in persuasive writing and editing.',
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # Define Task
            email_task = Task(
                description=task_description,
                agent=marketing_strategist,
                expected_output="Two final, polished versions of the cold email, ready for sending. Each version should be clearly distinct."
            )

            # Create the Crew
            email_crew = Crew(
                agents=[email_author, marketing_strategist, content_specialist],
                tasks=[email_task],
                verbose=True,
                process=Process.sequential
            )

            # Execution Flow
            st.info(f"🚀 Kicking off the email generation crew using {selected_model}...")
            with st.spinner("🤖 Agents are collaborating... Analyzing requirements, drafting, reviewing, and refining emails..."):
                emails_output = email_crew.kickoff()

            # Display Results
            st.success("✅ Crew finished generating emails!")
            st.subheader("Generated Emails Output:")
            st.markdown(emails_output)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your setup: API Key validity, network connection, task description, and library versions.")
            # Optional: print traceback to console for more detailed debugging
            import traceback
            traceback.print_exc()

else:
    st.info("Configure your Google API Key and task details in the sidebar, then click 'Generate Emails' to start.")

# --- Footer ---
st.markdown("---")
st.markdown("Powered by [CrewAI](https://github.com/joaomdmoura/crewAI), [Google Gemini](https://ai.google.dev/), and [Streamlit](https://streamlit.io)")
