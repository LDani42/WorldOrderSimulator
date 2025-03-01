"""
app.py

A Streamlit web application that:
• Fetches the top 5 politics-related news stories from News API.
• Lets you choose one of the headlines or enter a custom scenario.
• Provides multiple sliders (Global Cooperation, Tech Disruption, Public Sentiment, etc.).
• Makes TWO separate calls to GPT-4 (new openai interface):
   1) A "Best Case" scenario
   2) A "Worst Case" scenario
• Displays both scenarios side by side.

Environment Variables:
- OPENAI_API_KEY
- NEWSAPI_KEY

Usage:
1. Create a .env or set environment variables for the keys.
2. streamlit run app.py
"""

import os
import openai
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env (optional if you're using st.secrets or a hosting env)
load_dotenv()

# Retrieve API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]

def fetch_top_news():
    """
    Fetches the top 5 politics-related news stories from NewsAPI using 'everything' endpoint.
    Returns:
        (articles, error): articles = list of article dictionaries, error = string or None
    """
    url = "https://newsapi.org/v2/everything?q=politics&language=en&pageSize=5"
    headers = {"X-Api-Key": NEWSAPI_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None, f"Error: {response.status_code} - {response.text}"
    
    data = response.json()
    articles = data.get("articles", [])
    return articles, None

def get_scenario_response(
    scenario_type: str,
    news_text: str,
    cooperation_level: int,
    tech_disruption_level: int,
    public_sentiment_level: int,
    economic_volatility: int,
    environmental_stress: int
) -> str:
    """
    Makes a call to GPT-4 to generate a best or worst case scenario.

    Args:
        scenario_type (str): "best" or "worst"
        news_text (str): The chosen news scenario or custom scenario text
        cooperation_level, tech_disruption_level, etc.: integer sliders (0-100)

    Returns:
        content (str): GPT-4's response text
    """

    if scenario_type.lower() == "best":
        scenario_instruction = (
            "Focus on cooperative, optimistic paths where diplomatic efforts, "
            "strategic foresight, or altruistic motives lead to positive outcomes."
        )
        scenario_title = "BEST CASE SCENARIO"
    else:  # "worst"
        scenario_instruction = (
            "Focus on conflict-driven, opportunistic, or poorly managed paths "
            "where self-interest escalates tensions and leads to negative outcomes."
        )
        scenario_title = "WORST CASE SCENARIO"

    # Build the prompt
    prompt = f"""
You are a geopolitical analyst with expertise in game theory, historical precedents,
and environmental/economic/social factors. Below is a political news summary (or custom scenario) 
plus multiple parameters:

(1) Global Cooperation Level (0-100)
(2) Tech Disruption Level (0-100)
(3) Public Sentiment Level (0-100)
(4) Economic Volatility (0-100)
(5) Environmental Stress (0-100)

**Your task**:
- Generate a **{scenario_title}** spanning the next 5 years, divided into:
  - Short-term (6-12 months)
  - Mid-term (1-3 years)
  - Long-term (3-5 years)
- Show cause-effect (domino effect) reasoning, referencing relevant stakeholders (nations, leaders, organizations).
- Provide historical analogies if appropriate.
- {scenario_instruction}

Structure the output as:
**{scenario_title}**:
(A) Short-term:
(B) Mid-term:
(C) Long-term:
(D) Historical Analogies (if any)

News Text:
\"\"\"{news_text}\"\"\"

Parameter Values:
- Global Cooperation Level: {cooperation_level}
- Tech Disruption Level: {tech_disruption_level}
- Public Sentiment Level: {public_sentiment_level}
- Economic Volatility: {economic_volatility}
- Environmental Stress: {environmental_stress}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.2,   # High for more creativity
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        content = response.choices[0].message["content"].strip()
    except Exception as e:
        content = f"Error generating {scenario_type} scenario: {e}"

    return content

def main():
    st.title("Future Scenarios Simulator")
    st.markdown("**Generates two separate GPT-4 prompts (Best & Worst) and displays side by side.**")

    # --- Fetch top news ---
    st.subheader("Top 5 Current Politics News Stories")
    articles, error = fetch_top_news()
    if error:
        st.error(f"Failed to fetch news stories: {error}")
        st.stop()

    # Build a dictionary for story selection
    story_options = {}
    for idx, article in enumerate(articles):
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        content = article.get("content", "") or ""
        combined_text = f"{title}\n\nDescription: {description}\n\nContent: {content}"
        story_options[f"Article {idx+1}: {title}"] = combined_text

    story_options["Custom Scenario"] = "Enter your own custom scenario below."

    selection = st.radio("Select a politics news story or custom scenario:", list(story_options.keys()))
    if selection == "Custom Scenario":
        custom_text = st.text_area("Enter your custom scenario:", height=150)
        news_text = custom_text
    else:
        news_text = story_options[selection]

    st.markdown("---")

    # --- Sliders for parameters ---
    st.subheader("Scenario Parameters")
    cooperation_level = st.slider("Global Cooperation Level", 0, 100, 50)
    tech_disruption_level = st.slider("Tech Disruption Level", 0, 100, 50)
    public_sentiment_level = st.slider("Public Sentiment Level", 0, 100, 50)
    economic_volatility = st.slider("Economic Volatility", 0, 100, 50)
    environmental_stress = st.slider("Environmental Stress", 0, 100, 50)

    if st.button("Generate Best & Worst Scenarios"):
        if not news_text.strip():
            st.error("Please enter or select a scenario before generating.")
            return

        with st.spinner("Generating scenarios..."):
            # Call for BEST CASE
            best_case_output = get_scenario_response(
                scenario_type="best",
                news_text=news_text,
                cooperation_level=cooperation_level,
                tech_disruption_level=tech_disruption_level,
                public_sentiment_level=public_sentiment_level,
                economic_volatility=economic_volatility,
                environmental_stress=environmental_stress
            )

            # Call for WORST CASE
            worst_case_output = get_scenario_response(
                scenario_type="worst",
                news_text=news_text,
                cooperation_level=cooperation_level,
                tech_disruption_level=tech_disruption_level,
                public_sentiment_level=public_sentiment_level,
                economic_volatility=economic_volatility,
                environmental_stress=environmental_stress
            )

        # Display side by side
        st.markdown("## Scenario Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Best Case")
            st.markdown(best_case_output)
        with col2:
            st.markdown("### Worst Case")
            st.markdown(worst_case_output)

        st.markdown("---")
        st.caption("These are hypothetical scenarios generated by AI. Real events may differ. Source: [News API](https://newsapi.org).")

if __name__ == "__main__":
    main()
