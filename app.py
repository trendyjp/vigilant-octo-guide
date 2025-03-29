import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
# Required for ChatGoogleGenerativeAI
import google.generativeai as genai

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="CrewAI Email Generator (Gemini)", layout="wide")
st.title("🚀 CrewAI Cold Email Generator (using Google Gemini)")
st.markdown("""
Generate two variations of a cold email promoting a video editing solution using a team of AI agents powered by Google Gemini.
Enter your Google API Key and click 'Generate Emails'.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    # Get Google API Key from user
    google_api_key = st.text_input("Google API Key", type="password", key="google_api_key",
                                   help="Get yours from Google AI Studio: https://aistudio.google.com/app/apikey")

    # Option to customize the task description
    st.subheader("Task Customization")
    default_task_desc = '''1. Generate two distinct variations of a cold email promoting a video editing solution.
2. Evaluate the written emails for their effectiveness and engagement.
3. Scrutinize the emails for grammatical correctness and clarity.
4. Adjust the emails to align with best practices for cold outreach. Consider the feedback
provided to the marketing_strategist.
5. Revise the emails based on all feedback, creating two final versions.'''
    task_description = st.text_area("Email Task Description", value=default_task_desc, height=300, key="task_desc")

    # Model Selection (Optional - keeping gemini-pro for now)
    # model_options = ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest"] # Add more if needed
    # selected_model = st.selectbox("Select Gemini Model", model_options, index=0)
    selected_model = "gemini-pro" # Fixed for simplicity

    # Temperature Slider (Optional)
    # temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.5, 0.1)
    temperature = 0.5 # Fixed for simplicity

    # Button to trigger the process
    generate_button = st.button("Generate Emails", key="generate")

# --- Main App Logic ---
if generate_button:
    if not google_api_key:
        st.error("🚨 Please enter your Google API Key in the sidebar.")
    elif not task_description:
        st.error("🚨 Please enter a task description in the sidebar.")
    else:
        try:
            # Configure the Google Generative AI library (optional but good practice)
            # genai.configure(api_key=google_api_key) # The ChatGoogleGenerativeAI class handles this too

            # Initialize LLM inside the button click handler to use the provided API key
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                verbose=True, # Will print details in the console where Streamlit runs
                temperature=temperature,
                google_api_key=google_api_key # Pass the key directly
            )

            # Initialize Tools
            tool_search = DuckDuckGoSearchRun()

            # Define Agents (using the initialized LLM)
            email_author = Agent(
                role='Professional Email Author',
                goal='Craft concise and engaging emails based on the task description',
                backstory='Experienced in writing impactful marketing emails using current best practices.',
                verbose=True,
                allow_delegation=False,
                llm=llm,
                tools=[tool_search]
            )
            marketing_strategist = Agent(
                role='Marketing Strategist',
                goal='Lead the team in creating effective cold emails based on the provided task description',
                backstory='A seasoned Chief Marketing Officer with a keen eye for standout marketing content and strategy.',
                verbose=True,
                allow_delegation=True,
                llm=llm
            )
            content_specialist = Agent(
                role='Content Specialist',
                goal='Critique and refine email content for clarity, grammar, and persuasiveness',
                backstory='A professional copywriter with a wealth of experience in persuasive writing and editing.',
                verbose=True,
                allow_delegation=False,
                llm=llm
            )

            # Define Task using the description from the UI
            email_task = Task(
                description=task_description,
                agent=marketing_strategist,  # The Marketing Strategist leads
                expected_output="Two final, polished versions of the cold email, ready for sending. Each version should be clearly distinct."
            )

            # Create the Crew
            email_crew = Crew(
                agents=[email_author, marketing_strategist, content_specialist],
                tasks=[email_task],
                verbose=True, # Logs details to the console where Streamlit runs
                process=Process.sequential
            )

            # Execution Flow with status indicator
            st.info(f"🚀 Kicking off the email generation crew using {selected_model}... This might take a moment.")
            with st.spinner("🤖 Agents are collaborating... Analyzing requirements, drafting, reviewing, and refining emails..."):
                # Run the crew
                emails_output = email_crew.kickoff()

            # Display Results
            st.success("✅ Crew finished generating emails!")
            st.subheader("Generated Emails Output:")
            # Use markdown for potentially better formatting if the output is markdown-friendly
            st.markdown(emails_output)
            # Or use a text area for raw text output
            # st.text_area("Emails", value=emails_output, height=400)

        # Handle potential errors from Google API (like invalid key)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your Google API Key, ensure it's enabled for the Gemini API, and verify your network connection.")
            # You might want to print the full traceback to the console for debugging
            # import traceback
            # traceback.print_exc()

else:
    st.info("Configure your Google API Key and task details in the sidebar, then click 'Generate Emails' to start.")

# --- Optional: Add footer or more info ---
st.markdown("---")
st.markdown("Powered by [CrewAI](https://github.com/joaomdmoura/crewAI), [Google Gemini](https://ai.google.dev/), and [Streamlit](https://streamlit.io)")
