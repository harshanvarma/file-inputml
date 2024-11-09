from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import json

# Initialize Ollama LLM with Llama model
llm = Ollama(model="llama3:8b")

# Create a template specifically for Indian nutrition planning
indian_nutrition_template = """
You are an expert Indian nutritionist. Based on the following user information, provide a personalized Indian diet plan:

User Profile:
Height: {height} cm
Weight: {weight} kg
Age: {age}
Activity Level: {activity_level}
Goals: {goals}
Dietary Preferences: {dietary_preferences}
Health Issues: {health_issues}

Please provide a detailed Indian diet plan including:
1. Calculate daily calorie requirements
2. Provide macronutrient distribution (proteins, carbs, fats)
3. Suggest a full day meal plan with traditional Indian foods including:
   - Early morning (if applicable)
   - Breakfast
   - Mid-morning snack
   - Lunch
   - Evening snack
   - Dinner
4. Include specific Indian dishes and portion sizes
5. Mention which region each dish is from
6. Include spices and ingredients that have additional health benefits
7. Hydration recommendations including traditional Indian drinks

Previous conversation context:
{chat_history}

User Question: {user_input}

Guidelines for responses:
- Focus on traditional Indian ingredients and dishes
- Include regional varieties when possible
- Suggest healthy Indian alternatives to processed foods
- Include traditional wisdom about spices and herbs
- Provide measurements in Indian kitchen units (katori, cups, etc.)
"""

# Create prompt template
prompt = PromptTemplate(
    input_variables=[
        "height", "weight", "age", "activity_level", "goals", 
        "dietary_preferences", "health_issues", "chat_history", "user_input"
    ],
    template=indian_nutrition_template
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

def calculate_bmi(weight, height):
    """Calculate BMI and return category"""
    bmi = weight / ((height/100) ** 2)
    if bmi < 18.5:
        return f"BMI: {bmi:.1f} (Underweight)"
    elif bmi < 25:
        return f"BMI: {bmi:.1f} (Normal)"
    elif bmi < 30:
        return f"BMI: {bmi:.1f} (Overweight)"
    else:
        return f"BMI: {bmi:.1f} (Obese)"

def create_indian_nutrition_bot():
    st.title("🍛 Indian Nutrition Planning Assistant")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User Information Form with Indian context
    with st.sidebar:
        st.header("Your Information")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65)
        age = st.number_input("Age", min_value=15, max_value=100, value=30)
        
        activity_level = st.selectbox(
            "Activity Level",
            ["Sedentary (Office job, minimal exercise)",
             "Lightly Active (Light exercise 1-3 days/week)",
             "Moderately Active (Exercise 3-5 days/week)",
             "Very Active (Exercise 6-7 days/week)",
             "Extremely Active (Athletic training)"]
        )
        
        goals = st.selectbox(
            "Fitness Goals",
            ["Weight Loss",
             "Weight Gain",
             "Muscle Gain",
             "Maintenance",
             "Better Energy Levels",
             "Blood Sugar Management",
             "Heart Health"]
        )
        
        dietary_preferences = st.multiselect(
            "Dietary Preferences",
            ["Vegetarian", "Vegan", "Non-Vegetarian", 
             "Jain", "No Onion-Garlic",
             "South Indian", "North Indian", 
             "Gujarati", "Bengali", "Punjabi"],
            default=["Vegetarian"]
        )
        
        health_issues = st.multiselect(
            "Health Issues",
            ["None", "Diabetes", "Hypertension", 
             "Cholesterol", "Thyroid",
             "PCOS", "Lactose Intolerance",
             "Gluten Sensitivity"],
            default=["None"]
        )

        # Display BMI
        st.info(calculate_bmi(weight, height))

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("Ask about your Indian nutrition plan..."):
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
            "dietary_preferences": ", ".join(dietary_preferences),
            "health_issues": ", ".join(health_issues),
            "user_input": user_input
        })

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def save_chat_history(messages, filename="indian_nutrition_chat_history.json"):
    with open(filename, "w") as f:
        json.dump(messages, f)

def load_chat_history(filename="indian_nutrition_chat_history.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    create_indian_nutrition_bot()