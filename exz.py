from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st


llm = Ollama(
    model="mistral",
    temperature=0.7
)


nutrition_template = """
You are an expert Indian nutritionist and fitness consultant. Create a comprehensive nutrition and fitness plan based on the following user data:

PERSONAL INFORMATION:
- Age: {age}
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- Target Weight: {target_weight} kg
- Daily Sleep Hours: {sleep_hours}

HEALTH DATA:
- Medical Conditions: {medical_conditions}
- Allergies: {allergies}
- Current Medications: {medications}
- Blood Type: {blood_type}

FITNESS PROFILE:
- Primary Fitness Activities: {fitness_activities}
- Workout Frequency: {workout_frequency}
- Exercise Duration: {exercise_duration}
- Fitness Goals: {fitness_goals}

DIETARY INFORMATION:
- Dietary Preference: {diet_type}
- Food Allergies: {food_allergies}
- Meal Frequency: {meal_frequency}
- Preferred Cuisine: {preferred_cuisine}
- Budget Constraints: {budget_level}

Based on this information, provide a detailed response in the following format:

### MEAL SUGGESTIONS ###
1. Early Morning (Pre-workout):
2. Breakfast:
3. Mid-morning Snack:
4. Lunch:
5. Evening Snack:
6. Post-workout:
7. Dinner:

[Include specific Indian dishes, portion sizes, and timing]

### NUTRIENT ANALYSIS ###
1. Macronutrients Required:
   - Proteins:
   - Carbohydrates:
   - Fats:

2. Key Micronutrients:
   - Essential vitamins:
   - Minerals:
   - Other nutrients:

### FITNESS-SPECIFIC NUTRITION ###
[Provide specific nutrition recommendations based on their {fitness_activities} and {fitness_goals}]

### AFFORDABLE ALTERNATIVES ###
[List expensive nutrient sources and their cheaper Indian alternatives with similar nutritional value]

Keep recommendations focused on Indian foods and ingredients. Include regional dishes and seasonal considerations.
"""


prompt = PromptTemplate(
    input_variables=[
        "age", "gender", "height", "weight", "target_weight", "sleep_hours",
        "medical_conditions", "allergies", "medications", "blood_type",
        "fitness_activities", "workout_frequency", "exercise_duration", "fitness_goals",
        "diet_type", "food_allergies", "meal_frequency", "preferred_cuisine", "budget_level"
    ],
    template=nutrition_template
)

# Create LLM chain
nutrition_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

def main():
    st.title("🍛 Comprehensive Indian Nutrition & Fitness Planner")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
        target_weight = st.number_input("Target Weight (kg)", min_value=20, max_value=300, value=70)
        sleep_hours = st.slider("Daily Sleep Hours", min_value=4, max_value=12, value=7, step=1)
        
        st.subheader("Health Information")
        medical_conditions = st.multiselect(
            "Medical Conditions",
            ["None", "Diabetes", "Hypertension", "Thyroid", "PCOS", "Heart Disease", 
             "Kidney Issues", "Liver Problems", "Other"],
            default=["None"]
        )
        allergies = st.text_input("Allergies", "None")
        medications = st.text_input("Current Medications", "None")
        blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-", "Don't Know"])

    with col2:
        st.subheader("Fitness Profile")
        fitness_activities = st.multiselect(
            "Fitness Activities",
            ["Gym/Weight Training", "Yoga", "Swimming", "Running", "Cycling", 
             "Sports", "Walking", "HIIT", "Pilates", "Dancing", "Martial Arts"],
            default=["Walking"]
        )
        workout_frequency = st.selectbox(
            "Workout Frequency",
            ["1-2 times/week", "3-4 times/week", "5-6 times/week", "Daily", "Multiple times per day"]
        )
        exercise_duration = st.selectbox(
            "Exercise Duration",
            ["15-30 minutes", "30-45 minutes", "45-60 minutes", "60-90 minutes", "90+ minutes"]
        )
        fitness_goals = st.multiselect(
            "Fitness Goals",
            ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility", 
             "Strength", "General Fitness", "Sports Performance"],
            default=["General Fitness"]
        )

    with col3:
        st.subheader("Dietary Information")
        diet_type = st.selectbox(
            "Dietary Preference",
            ["Vegetarian", "Vegan", "Non-vegetarian", "Eggetarian", 
             "Jain", "No Onion-Garlic"]
        )
        food_allergies = st.multiselect(
            "Food Allergies/Intolerances",
            ["None", "Dairy", "Gluten", "Nuts", "Soy", "Shellfish", "Other"],
            default=["None"]
        )
        meal_frequency = st.selectbox(
            "Preferred Meal Frequency",
            ["2 meals", "3 meals", "4 meals", "5 meals", "6+ meals"]
        )
        preferred_cuisine = st.multiselect(
            "Preferred Regional Cuisine",
            ["North Indian", "South Indian", "Bengali", "Gujarati", 
             "Maharashtrian", "Punjabi", "Kerala", "Any"],
            default=["Any"]
        )
        budget_level = st.select_slider(
            "Budget Level",
            options=["Very Limited", "Limited", "Moderate", "Flexible", "Unlimited"],
            value="Moderate"
        )

    # Generate button
    if st.button("Generate Nutrition Plan"):
        with st.spinner("Creating your personalized nutrition plan..."):
            # Generate plan
            response = nutrition_chain.run(
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                target_weight=target_weight,
                sleep_hours=sleep_hours,
                medical_conditions=", ".join(medical_conditions),
                allergies=allergies,
                medications=medications,
                blood_type=blood_type,
                fitness_activities=", ".join(fitness_activities),
                workout_frequency=workout_frequency,
                exercise_duration=exercise_duration,
                fitness_goals=", ".join(fitness_goals),
                diet_type=diet_type,
                food_allergies=", ".join(food_allergies),
                meal_frequency=meal_frequency,
                preferred_cuisine=", ".join(preferred_cuisine),
                budget_level=budget_level
            )
            
            # Display response
            st.markdown(response)

if __name__ == "__main__":
    main()