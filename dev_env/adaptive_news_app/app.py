import streamlit as st
from sample_news_data import news_data

st.set_page_config(page_title="Adaptive News Recommender", layout="centered")

# Title
st.title(" AI-Based Adaptive News Recommender")

# Intro text
st.markdown("""
This adaptive system tailors the news feed based on your selected interests.
Choose the categories you care about, and the content will update in real-time.
""")

# Initialize session state
if "preferences" not in st.session_state:
    st.session_state.preferences = []

# Sidebar for user preferences
st.sidebar.header("Select Your News Interests")
categories = list(news_data.keys())
selected_categories = st.sidebar.multiselect(
    "Choose topics to personalize your feed:",
    options=categories,
    default=st.session_state.preferences
)

# Save preferences to session state
st.session_state.preferences = selected_categories

# Show selected preferences
st.write("### Your selected topics:")
if not selected_categories:
    st.info("Please select at least one topic to see recommended news.")
else:
    st.success(", ".join(selected_categories))

# Adaptive content display
st.write("### Recommended News Based on Your Interests:")
for category in selected_categories:
    st.subheader(f" {category}")
    for headline in news_data[category]:
        st.markdown(f"- {headline}")
