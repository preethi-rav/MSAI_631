# 📰 AI-Based Adaptive News Recommender

This project showcases an **AI-based Adaptive User Interface (AUI)** developed using **Python** and **Streamlit**. It is a simple yet effective simulation of a personalized news application that **adapts content based on user-selected interests** in real-time. The project is part of an academic assignment focused on exploring **Human-Computer Interaction (HCI)** and **AI-powered interfaces**.

---

## 📌 Project Objective

The goal of this project is to demonstrate:

- **Adaptive User Interfaces (AUIs)** that change content/layout based on user input.
- Basic principles of **Intelligent User Interfaces (IUIs)** using rule-based AI logic.
- Real-time content personalization using Python and Streamlit.
- The relationship between user behavior and interface adaptation in AI-based HCI systems.

---

## 🧠 How It Works

1. The user selects preferred news categories from a sidebar (e.g., Technology, Health, Sports).
2. The system filters and displays only the headlines relevant to the selected interests.
3. The interface updates **in real-time** as the user changes their preferences.
4. Session state is preserved to simulate persistent user modeling.

While this demo uses **mock data** and **rule-based logic**, it effectively illustrates how modern interfaces can adapt to individual users using AI principles.

---

## 🎯 Key Features

- 🔄 Adaptive content based on real-time user selections
- 🧠 Simulated intelligent recommendation using rule-based filtering
- 🖥️ Clean, responsive web interface using Streamlit
- 📂 Simple and modular codebase, easy to understand and extend
- 💬 Explanatory UI for users unfamiliar with adaptive systems

---

## 📁 Folder Structure

adaptive-news-recommender/
│
├── app.py # Main Streamlit application
├── sample_news_data.py # Static news content categorized by topic
├── requirements.txt # Dependencies needed to run the app
└── README.md # Project instructions and documentation

yaml
Copy code

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/adaptive-news-recommender.git
cd adaptive-news-recommender
Replace the GitHub link above with your actual repository URL.

### 2. Set Up a Virtual Environment (Optional but Recommended)

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Dependencies

bash
Copy code
pip install -r requirements.txt
This will install streamlit==1.30.0 or the latest compatible version.

### 4. Run the Application

bash
Copy code
streamlit run app.py
Open the provided local address (usually http://localhost:8501) in your browser to interact with the app.

AI Tools Used

The following AI-powered tools assisted in the creation of this project:

ChatGPT – For generating and refining code and documentation.

GitHub Copilot – For inline coding assistance in the IDE.

Google Bard – Used for brief concept clarification on adaptive UIs.

All outputs were reviewed and manually verified before inclusion.