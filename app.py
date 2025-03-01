"""
app.py

This Streamlit app:
• Fetches the top 5 politics-related news stories from News API.
• Lets you choose a headline or enter a custom scenario.
• Provides multiple sliders (Global Cooperation, Tech Disruption, Public Sentiment, etc.).
• Makes TWO separate calls to GPT-4:
   1) A "Best Case" scenario prompt
   2) A "Worst Case" scenario prompt
• Displays both scenarios side by side.

Installation/Usage:
1. .env file with:
   OPENAI_API_KEY=your_openai_key
   NEWSAPI_KEY=your_newsapi_key
2. pip install streamlit openai requests python-dotenv
3. streamlit run app.py
"""

import os
from dotenv import load_dotenv
import streamlit as st
import requests
import openai
from openai import ChatCompletion

# Load environment variables
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

def fetch_top_news():
    """
    Fetches the top 5 politics-related news stories from the News API.
    Returns:
        (articles, error): articles = list of article dicts, error = error message or None.
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
    scenario_type,
    news_text,
    cooperation_level,
    tech_disruption_level,
    public_sentiment_level,
    economic_volatility,
    environmental_stress
):
    """
    Makes a call to GPT-4 to generate EITHER a best-case OR worst-case scenario
    over short-, mid-, and long-term, using a specialized prompt for each scenario_type.
    """
    # Scenario-specific instructions
    if scenario_type.lower() == "best":
        scenario_instruction = "Focus on socially optimal, cooperative paths, highlighting how collaboration, diplomacy, or strategic foresight can lead to positive outcomes."
    else:  # worst
        scenario_instruction = "Focus on the opportunistic, conflict-driven, or poorly managed paths where self-interest and escalation lead to negative outcomes."
    
    # Build the prompt
    prompt = f"""
You are a geopolitical analyst with deep expertise in:
- Game theory
- Historical precedents
- Environmental, social, and economic factors

Below is a political news summary (or custom scenario) plus multiple parameters:
1. Global Cooperation Level (0–100)
2. Tech Disruption Level (0–100)
3. Public Sentiment Level (0–100)
4. Economic Volatility (0–100)
5. Environmental Stress (0–100)

**Your goal**:
- Generate a **{scenario_type.upper()} CASE** scenario spanning the next 5 years, divided into:
  - Short-term (6-12 months)
  - Mid-term (1-3 years)
  - Long-term (3-5 years)
- Incorporate cause-effect (domino effect) reasoning, game-theory logic, historical analogies, and relevant stakeholders (nations, leaders, organizations).
- {scenario_instruction}
- Show how changes in Tech Disruption, Public Sentiment, Economic Volatility, Environmental Stress, and Global Cooperation shape events and alliances.

Structure your output as:
**{scenario_type.capitalize()} Case Scenario**:
(A) Short-term:
(B) Mid-term:
(C) Long-term:
(D) Relevant Historical Analogies

Do not provide extra disclaimers. Keep it focused on the {scenario_type.upper()} scenario.

News Text:
\"\"\"{news_text}\"\"\"

Parameter Values:
- Global Cooperation Level: {cooperation_level}
- Tech Disruption Level: {tech_disruption_level}
- Public Sentiment Level: {public_sentiment_level}
- Economic Volatility: {economic_volatility}
- Environmental Stress: {environmental_stress}
"""

    # Call OpenAI
    try:
        response = ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.2,   # High temperature for more creativity
            max_tokens=1500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        content = response.choices[0].message['content'].strip()
    except Exception as e:
        content = f"Error generating {scenario_type} scenario: {e}"
    
    return content


def main():
    st.title("Two-Prompt Future Scenarios (Best & Worst)")

    # Fetch top politics news
    st.subheader("Top 5 Current Politics News Stories")
    articles, error = fetch_top_news()
    if error:
        st.error("Failed to fetch news stories. Please check your NEWSAPI_KEY or try again later.")
        st.stop()
    
    # Build story selection
    story_options = {}
    for idx, article in enumerate(articles):
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        content = article.get("content", "") or ""
        
        combined_text = f"{title}\n\nDescription: {description}\n\nContent: {content}"
        story_options[f"Article {idx+1}: {title}"] = combined_text
    
    # Custom scenario
    story_options["Custom Scenario"] = "Enter your own custom scenario below."
    
    selection = st.radio("Select a politics news story or choose a custom scenario:", list(story_options.keys()))
    
    if selection == "Custom Scenario":
        custom_text = st.text_area("Enter your custom scenario:", height=150)
        news_text = custom_text
    else:
        news_text = story_options[selection]
    
    st.markdown("---")
    
    st.subheader("Scenario Parameters")
    cooperation_level = st.slider("Global Cooperation Level", 0, 100, 50)
    tech_disruption_level = st.slider("Tech Disruption Level", 0, 100, 50)
    public_sentiment_level = st.slider("Public Sentiment Level", 0, 100, 50)
    economic_volatility = st.slider("Economic Volatility", 0, 100, 50)
    environmental_stress = st.slider("Environmental Stress", 0, 100, 50)
    
    if st.button("Generate Best & Worst Scenarios"):
        if not news_text.strip():
            st.error("Please provide a news story or scenario text.")
            return
        
        with st.spinner("Generating scenarios..."):
            # 1) BEST CASE PROMPT
            best_case_output = get_scenario_response(
                scenario_type="best",
                news_text=news_text,
                cooperation_level=cooperation_level,
                tech_disruption_level=tech_disruption_level,
                public_sentiment_level=public_sentiment_level,
                economic_volatility=economic_volatility,
                environmental_stress=environmental_stress
            )
            
            # 2) WORST CASE PROMPT
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
        st.markdown("## Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Best Case")
            st.markdown(best_case_output)
        with col2:
            st.markdown("### Worst Case")
            st.markdown(worst_case_output)
        
        st.markdown("---")
        st.caption("Hypothetical scenarios generated by AI. Real events may differ. Data source: [News API](https://newsapi.org).")

if __name__ == "__main__":
    main()
