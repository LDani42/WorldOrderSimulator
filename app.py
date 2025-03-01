import os
import time
import queue
import threading
import openai
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

###############################################################################
# 1) Default to WIDE LAYOUT and rename the app to something more engaging
###############################################################################
st.set_page_config(layout="wide")

def fetch_top_news():
    """
    Fetches the top 5 headlines in the 'general' category for the US,
    with language='en' (though top-headlines may ignore it).
    """
    url = (
        "https://newsapi.org/v2/top-headlines?"
        "country=us&"
        "category=general&"
        "language=en&"
        "pageSize=5"
    )
    headers = {"X-Api-Key": NEWSAPI_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, f"Error: {response.status_code} - {response.text}"
    
    data = response.json()
    articles = data.get("articles", [])
    return articles, None

###############################################################################
# 2) Enhanced Prompt with the 8-step scenario planning, referencing formatting
###############################################################################
def get_scenario_response_stream(
    scenario_type: str,
    news_text: str,
    cooperation_level: int,
    tech_disruption_level: int,
    public_sentiment_level: int,
    economic_volatility: int,
    environmental_stress: int
):
    """
    Streams a GPT-4.5-preview response for either a Best or Worst Case scenario.
    Yields chunks of content as they arrive from the API.
    """

    # Distinguish best/worst scenario instructions
    if scenario_type.lower() == "best":
        scenario_instruction = (
            "Focus on cooperative, optimistic paths where thoughtful diplomacy, "
            "strategic foresight, or altruistic motives lead to positive outcomes."
        )
        scenario_title = "BEST CASE SCENARIO"
    else:
        scenario_instruction = (
            "Focus on conflict-driven, opportunistic, or poorly managed paths "
            "where self-interest escalates tensions and leads to negative outcomes."
        )
        scenario_title = "WORST CASE SCENARIO"

    # Incorporate your 8-step logic guidance + structured, visually rich output
    prompt = f"""
You are an advanced political scientist and policy analyst specialized in scenario planning,
systems thinking, and strategic wargaming. We have a 'trigger event' (the news headline or 
custom scenario), and the following parameters:

(1) Global Cooperation Level (0-100)
(2) Tech Disruption Level (0-100)
(3) Public Sentiment Level (0-100)
(4) Economic Volatility (0-100)
(5) Environmental Stress (0-100)

**Important Guidance**:
- Provide a **{scenario_title}** spanning the next 5 years, divided into:
  - Short-term (6-12 months)
  - Mid-term (1-3 years)
  - Long-term (3-5 years)
- Incorporate the following structured framework, aligning with scenario planning 
  and systems thinking:
  1. Define the Scope & Trigger Event
  2. Establish the Context
  3. Identify Key Actors & Their Relationships
  4. Determine System Structure & Dependencies
  5. Specify Simulation Rules & Propagation Mechanisms
  6. Develop Possible Scenarios (Here: the specific {scenario_title})
  7. Analyze Outcomes & Policy Implications
  8. Communicate Findings & Next Steps
- Make the result visually appealing and highly readable using:
  - Headers (#, ##, ###)
  - Subheadings
  - Bullet points, numbered lists
  - Tables (Markdown tables)
  - ASCII art/diagrams (for causal links, flows, or relationships)
- {scenario_instruction}

**Structure your final output** as you see fit (you may use headings that correspond to the 8 steps),
but ensure to include at least:
1. A top-level heading for {scenario_title} (with a brief intro).
2. A breakdown by time horizon:
   - Short-term (6-12 months)
   - Mid-term (1-3 years)
   - Long-term (3-5 years)
3. Historical or theoretical analogies if relevant.
4. One or more ASCII diagrams or tables if relevant to illustrate important points.

**News Text / Trigger Event**:
\"\"\"{news_text}\"\"\"

**Parameter Values**:
- Global Cooperation Level: {cooperation_level}
- Tech Disruption Level: {tech_disruption_level}
- Public Sentiment Level: {public_sentiment_level}
- Economic Volatility: {economic_volatility}
- Environmental Stress: {environmental_stress}
""".strip()

    try:
        # Use gpt-4.5-preview model
        response = openai.ChatCompletion.create(
            model="o1",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="medium",   # High for creative variety
            #temperature=1.2,   # High for creative variety
            max_completion_tokens=15000,   # Increase token limit significantly
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True        # Enable streaming
        )
        
        # Yield the content chunks as they arrive
        for chunk in response:
            chunk_content = chunk["choices"][0].get("delta", {}).get("content", "")
            if chunk_content:
                yield chunk_content

    except Exception as e:
        yield f"\n\nError generating {scenario_type} scenario: {e}"

###############################################################################
# 3) Threading & Queues for concurrent streaming of Best & Worst
###############################################################################
def stream_scenario_in_thread(
    out_queue: queue.Queue,
    scenario_type: str,
    news_text: str,
    cooperation_level: int,
    tech_disruption_level: int,
    public_sentiment_level: int,
    economic_volatility: int,
    environmental_stress: int
):
    """
    Run the streaming OpenAI call in a separate thread.
    Place the chunks into a queue. 
    End with None to signal completion.
    """
    try:
        for chunk in get_scenario_response_stream(
            scenario_type,
            news_text,
            cooperation_level,
            tech_disruption_level,
            public_sentiment_level,
            economic_volatility,
            environmental_stress
        ):
            out_queue.put(chunk)
    finally:
        # Signal that streaming is done
        out_queue.put(None)

###############################################################################
# 4) Main Streamlit App
###############################################################################
def main():
    # Updated title & introduction
    st.title("Crisis Cascade Simulator")
    st.markdown(
        """
        **Welcome to the Crisis Cascade Simulator!**  
        This tool helps you explore how a chosen “trigger event” (headline or custom scenario) 
        might unfold over the next five years. Leveraging scenario planning, systems thinking, 
        and strategic wargaming techniques, we’ll generate two parallel forecasts:  
        - **Best Case Scenario**  
        - **Worst Case Scenario**  

        Each scenario is streamed in real-time side-by-side, giving you a quick comparative 
        view of potential futures. Let’s dive in!
        """
    )

    # --- Fetch top news ---
    st.subheader("Top 5 US General News Stories")
    articles, error = fetch_top_news()
    if error:
        st.error(f"Failed to fetch news stories: {error}")
        st.stop()

    # Build story selection dictionary
    story_options = {}
    for idx, article in enumerate(articles):
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        content = article.get("content", "") or ""
        article_url = article.get("url", "") or ""
        
        combined_text = f"{title}\n\nDescription: {description}\n\nContent: {content}"
        label = f"Article {idx+1}: {title}"
        
        story_options[label] = {
            "text": combined_text,
            "link": article_url
        }

    # Add a custom scenario option
    story_options["Custom Scenario"] = {
        "text": "Enter your own custom scenario below.",
        "link": ""
    }

    # Selection radio
    selection = st.radio("Select a news story or custom scenario:", list(story_options.keys()))
    selected_story_data = story_options[selection]

    # If custom scenario is selected, let user input text
    if selection == "Custom Scenario":
        custom_text = st.text_area("Enter your custom scenario:", height=150)
        news_text = custom_text
    else:
        if selected_story_data["link"]:
            st.markdown(
                f'<a href="{selected_story_data["link"]}" target="_blank">Read Full Article</a>',
                unsafe_allow_html=True
            )
        news_text = selected_story_data["text"]

    st.markdown("---")

    # --- Sliders for parameters ---
    st.subheader("Scenario Parameters")
    cooperation_level = st.slider("Global Cooperation Level", 0, 100, 50)
    tech_disruption_level = st.slider("Tech Disruption Level", 0, 100, 50)
    public_sentiment_level = st.slider("Public Sentiment Level", 0, 100, 50)
    economic_volatility = st.slider("Economic Volatility", 0, 100, 50)
    environmental_stress = st.slider("Environmental Stress", 0, 100, 50)

    # Button to generate scenarios
    if st.button("Generate Best & Worst Scenarios"):
        if not news_text.strip():
            st.error("Please enter or select a scenario before generating.")
            return

        st.markdown("## Scenario Results")

        # Create two columns for side-by-side presentation
        col1, col2 = st.columns(2)

        # Prepare placeholders for streaming text
        with col1:
            st.markdown("### Best Case Scenario")
            best_placeholder = st.empty()
        with col2:
            st.markdown("### Worst Case Scenario")
            worst_placeholder = st.empty()

        # Queues to receive streamed text
        best_queue = queue.Queue()
        worst_queue = queue.Queue()

        # Threads for concurrency
        best_thread = threading.Thread(
            target=stream_scenario_in_thread,
            args=(
                best_queue,
                "best",
                news_text,
                cooperation_level,
                tech_disruption_level,
                public_sentiment_level,
                economic_volatility,
                environmental_stress
            ),
            daemon=True
        )
        worst_thread = threading.Thread(
            target=stream_scenario_in_thread,
            args=(
                worst_queue,
                "worst",
                news_text,
                cooperation_level,
                tech_disruption_level,
                public_sentiment_level,
                economic_volatility,
                environmental_stress
            ),
            daemon=True
        )

        # Start streaming threads
        best_thread.start()
        worst_thread.start()

        best_done = False
        worst_done = False
        best_accumulated = ""
        worst_accumulated = ""

        # Continuously read from both queues until both are done
        while not (best_done and worst_done):
            # Best scenario updates
            if not best_done:
                try:
                    chunk = best_queue.get_nowait()
                    if chunk is None:
                        best_done = True
                    else:
                        best_accumulated += chunk
                        best_placeholder.markdown(best_accumulated)
                except queue.Empty:
                    pass  # No chunk available this loop

            # Worst scenario updates
            if not worst_done:
                try:
                    chunk = worst_queue.get_nowait()
                    if chunk is None:
                        worst_done = True
                    else:
                        worst_accumulated += chunk
                        worst_placeholder.markdown(worst_accumulated)
                except queue.Empty:
                    pass  # No chunk available this loop

            time.sleep(0.05)  # short pause to avoid busy waiting

        # Join threads to ensure they're finished
        best_thread.join()
        worst_thread.join()

        st.markdown("---")
        st.caption(
            "These scenarios are hypothetical and generated by AI. "
            "Real events may differ. Source: [News API](https://newsapi.org)."
        )

if __name__ == "__main__":
    main()
