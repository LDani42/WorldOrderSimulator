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
    Streams a GPT-4 response for either a Best or Worst Case scenario.
    Yields chunks of content as they arrive from the API.
    """

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

    # Enhanced prompt to ensure realism, depth, and visual cues
    prompt = f"""
You are a geopolitical analyst with advanced degrees in political science, economics, 
and environmental studies. You have worked with global think-tanks on forecasting. 
Below is a news summary (or custom scenario) plus multiple parameters:

(1) Global Cooperation Level (0-100)
(2) Tech Disruption Level (0-100)
(3) Public Sentiment Level (0-100)
(4) Economic Volatility (0-100)
(5) Environmental Stress (0-100)

**Important Guidance**:
- Provide a {scenario_title} spanning the next 5 years, divided into:
  - Short-term (6-12 months)
  - Mid-term (1-3 years)
  - Long-term (3-5 years)
- Use realistic, data-informed reasoning based on the scenario. 
  For instance, if this is a local sports story, do NOT escalate it to global upheaval.
- Show cause-effect or domino-effect reasoning, referencing relevant stakeholders 
  (nations, leaders, organizations) only if it makes sense in the specific context.
- Where appropriate, include ASCII-based visual cues (e.g., tree diagrams, flow charts)
  to illustrate causal or domino effects.
- Provide historical analogies if they genuinely fit.
- {scenario_instruction}

**Structure your output** as:
**{scenario_title}**:
(A) Short-term (6-12 months):
(B) Mid-term (1-3 years):
(C) Long-term (3-5 years):
(D) Historical Analogies (if any):
(E) Visual Diagram (if helpful, in ASCII form)

News Text:
\"\"\"{news_text}\"\"\"

Parameter Values:
- Global Cooperation Level: {cooperation_level}
- Tech Disruption Level: {tech_disruption_level}
- Public Sentiment Level: {public_sentiment_level}
- Economic Volatility: {economic_volatility}
- Environmental Stress: {environmental_stress}
""".strip()

    try:
        # Stream the GPT-4 completion
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.2,   # High for creative variety
            max_tokens=7000,   # Increase token limit significantly
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
        # If there's an error, yield the error message
        yield f"\n\nError generating {scenario_type} scenario: {e}"

def main():
    st.title("Future Scenarios Simulator (Top US Headlines)")
    st.markdown("**Generates two GPT-4 streaming responses (Best & Worst) for each selected headline.**")

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

    if st.button("Generate Best & Worst Scenarios"):
        if not news_text.strip():
            st.error("Please enter or select a scenario before generating.")
            return

        st.markdown("## Scenario Results")

        # Create two columns for side-by-side streaming
        col1, col2 = st.columns(2)

        # --- Best Case Streaming ---
        with col1:
            st.markdown("### Best Case")
            best_case_placeholder = st.empty()
            best_text_accumulated = ""

            with st.spinner("Generating BEST CASE scenario..."):
                for chunk in get_scenario_response_stream(
                    scenario_type="best",
                    news_text=news_text,
                    cooperation_level=cooperation_level,
                    tech_disruption_level=tech_disruption_level,
                    public_sentiment_level=public_sentiment_level,
                    economic_volatility=economic_volatility,
                    environmental_stress=environmental_stress
                ):
                    best_text_accumulated += chunk
                    best_case_placeholder.markdown(best_text_accumulated)

        # --- Worst Case Streaming ---
        with col2:
            st.markdown("### Worst Case")
            worst_case_placeholder = st.empty()
            worst_text_accumulated = ""

            with st.spinner("Generating WORST CASE scenario..."):
                for chunk in get_scenario_response_stream(
                    scenario_type="worst",
                    news_text=news_text,
                    cooperation_level=cooperation_level,
                    tech_disruption_level=tech_disruption_level,
                    public_sentiment_level=public_sentiment_level,
                    economic_volatility=economic_volatility,
                    environmental_stress=environmental_stress
                ):
                    worst_text_accumulated += chunk
                    worst_case_placeholder.markdown(worst_text_accumulated)

        st.markdown("---")
        st.caption(
            "These scenarios are hypothetical and generated by AI. "
            "Real events may differ. Source: [News API](https://newsapi.org)."
        )

if __name__ == "__main__":
    main()
