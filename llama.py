from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import json

# Initialize Ollama with Llama 2
llm = Ollama(model="llama2")

# Create a template for generating nutrition plans
nutrition_template = """
You are a professional nutritionist. Based on the following user information, create a personalized nutrition plan:

User Information:
- Age: {age}
- Weight: {weight} kg
- Height: {height} cm
- Activity Level: {activity_level}
- Dietary Restrictions: {dietary_restrictions}
- Goals: {goals}

Please provide:
1. Daily caloric needs
2. Macronutrient distribution
3. Meal plan suggestions
4. Specific food recommendations
5. Supplements if needed

Previous conversation context:
{chat_history}

User Query: {user_input}

Please provide a detailed and personalized response:
"""

# Create prompt template
prompt = PromptTemplate(
    input_variables=["age", "weight", "height", "activity_level", 
                    "dietary_restrictions", "goals", "chat_history", "user_input"],
    template=nutrition_template
)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

# Create LLM chain
nutrition_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

def save_user_data(user_data):
    """Save user data to a JSON file"""
    with open('user_data.json', 'w') as f:
        json.dump(user_data, f)

def load_user_data():
    """Load user data from JSON file"""
    try:
        with open('user_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Streamlit UI
def main():
    st.title("Personalized Nutrition Planner")
    
    # Sidebar for user information
    st.sidebar.header("User Information")
    
    # Load saved user data
    user_data = load_user_data()
    
    if user_data is None:
        user_data = {
            "age": 30,
            "weight": 70,
            "height": 170,
            "activity_level": "Moderate",
            "dietary_restrictions": "None",
            "goals": "Weight maintenance"
        }
    
    # User input fields
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=user_data["age"])
    weight = st.sidebar.number_input("Weight (kg)", min_value=20, max_value=300, value=user_data["weight"])
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=user_data["height"])
    
    activity_level = st.sidebar.selectbox(
        "Activity Level",
        ["Sedentary", "Light", "Moderate", "Very Active", "Extremely Active"],
        index=["Sedentary", "Light", "Moderate", "Very Active", "Extremely Active"].index(user_data["activity_level"])
    )
    
    dietary_restrictions = st.sidebar.text_input("Dietary Restrictions", value=user_data["dietary_restrictions"])
    goals = st.sidebar.text_input("Fitness/Health Goals", value=user_data["goals"])
    
    # Save button for user data
    if st.sidebar.button("Save User Information"):
        user_data = {
            "age": age,
            "weight": weight,
            "height": height,
            "activity_level": activity_level,
            "dietary_restrictions": dietary_restrictions,
            "goals": goals
        }
        save_user_data(user_data)
        st.sidebar.success("User information saved!")

    # Main chat interface
    st.header("Chat with Your Nutrition Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your nutrition plan"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        response = nutrition_chain.run(
            age=age,
            weight=weight,
            height=height,
            activity_level=activity_level,
            dietary_restrictions=dietary_restrictions,
            goals=goals,
            user_input=prompt
        )

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()