from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import json

# Initialize Ollama LLM
llm = Ollama(model="mistral")

# Create a template for collecting user information
user_info_template = """
Please provide your nutrition plan recommendations based on the following user information:
Height: {height} cm
Weight: {weight} kg
Age: {age}
Activity Level: {activity_level}
Goals: {goals}
Dietary Restrictions: {dietary_restrictions}
Previous Health Issues: {health_issues}

Please provide a detailed daily nutrition plan including:
1. Total daily calorie requirement
2. Macronutrient breakdown
3. Meal timing recommendations
4. Specific food suggestions for each meal
5. Supplements if needed
6. Hydration recommendations

Previous conversation context:
{chat_history}

User Question: {user_input}
"""

# Create prompt template
prompt = PromptTemplate(
    input_variables=["height", "weight", "age", "activity_level", "goals", 
                    "dietary_restrictions", "health_issues", "chat_history", "user_input"],
    template=user_info_template
)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

# Create LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def create_nutrition_bot():
    st.title("🥗 Nutrition Planning Assistant")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User Information Form
    with st.sidebar:
        st.header("Your Information")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        age = st.number_input("Age", min_value=15, max_value=100, value=30)
        activity_level = st.selectbox(
            "Activity Level",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"]
        )
        goals = st.text_area("Fitness Goals", "e.g., weight loss, muscle gain, maintenance")
        dietary_restrictions = st.text_area("Dietary Restrictions", "e.g., vegetarian, gluten-free, none")
        health_issues = st.text_area("Health Issues", "e.g., diabetes, hypertension, none")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask about your nutrition plan..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        response = chain.run({
            "height": height,
            "weight": weight,
            "age": age,
            "activity_level": activity_level,
            "goals": goals,
            "dietary_restrictions": dietary_restrictions,
            "health_issues": health_issues,
            "user_input": user_input
        })

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def save_chat_history(messages, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(messages, f)

def load_chat_history(filename="chat_history.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    create_nutrition_bot()