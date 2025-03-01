"""
app.py

This Streamlit web application:
• Fetches the top 5 headlines (category=politics, country=us, language=en) from News API.
• Displays each story's title, plus a clickable link to the original article.
• Lets you choose one of the headlines or enter a custom scenario.
• Provides multiple sliders (Global Cooperation, Tech Disruption, Public Sentiment, etc.).
• Makes TWO separate calls to GPT-4 (new openai interface):
   1) A "Best Case" scenario
   2) A "Worst Case" scenario
• Displays both scenarios side by side.
"""

import os
import openai
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def fetch_top_news():
    """
    Fetches the top 5 headlines in the 'politics' category for the US, 
    optionally with language set to 'en' (though top-headlines may ignore it).
    """
    url = (
        "https://newsapi.org/v2/top-headlines?"
        "country=us&"
        "category=politics&"
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
    """

    if scenario_type.lower() == "best":
        scenario_instruction = (
            "Focus on cooperative, optimistic paths where diplomatic efforts, "
            "strategic foresight, or altruistic motives lead to positive outcomes."
        )
        scenario_title = "BEST CASE SCENARIO"
    else:
        scenario_instruction = (
            "Focus on conflict-driven, opportunistic, or poorly managed paths "
            "where self-interest escalates tensions and leads to negative outcomes."
        )
        scenario_title = "WORST CASE SCENARIO"

    prompt = f"""
You are a geopolitical analyst with expertise in game theory, historical precedents,
and environmental/economic/social factors. Below is a news summary (or custom scenario) 
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
    st.title("Future Scenarios Simulator (Top US Headlines)")
    st.markdown("**Gets the most recent News Stories and Reasons (Best & Worst) Scenarios for their Impact on the Global Order.**")

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

    selection = st.radio("Select a news story or custom scenario:", list(story_options.keys()))
    selected_story_data = story_options[selection]

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

    if st.button("Generate Best & Worst Scenarios"):
        if not news_text.strip():
            st.error("Please enter or select a scenario before generating.")
            return

        with st.spinner("Generating scenarios..."):
            # BEST CASE
            best_case_output = get_scenario_response(
                scenario_type="best",
                news_text=news_text,
                cooperation_level=cooperation_level,
                tech_disruption_level=tech_disruption_level,
                public_sentiment_level=public_sentiment_level,
                economic_volatility=economic_volatility,
                environmental_stress=environmental_stress
            )

            # WORST CASE
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
