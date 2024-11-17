import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# Ensure your GenAI (Gemini) setup uses OpenAI API
import google.generativeai as genai
from annotated_text import annotated_text, annotation

# Categories for predictions
categories = ["healthy", "powdery", "rust"]

# Load models (adjust paths to match your directory)


@st.cache_resource
def load_models():
    models = {
        "CNN": load_model("models/cnn_model.keras"),
        "VGG": load_model("models/vgg_model.keras"),
        "Inception": load_model("models/inception_model.keras"),
        "Xception": load_model("models/xception_model.keras"),
    }
    return models


models = load_models()

# Preprocessing the input image


def prepare_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict using all models and average the probabilities


def predict_with_models(image):
    probabilities = []
    for model in models.values():
        preds = model.predict(image)[0]
        probabilities.append(preds)
    avg_probabilities = np.mean(probabilities, axis=0)
    return avg_probabilities

# Fuzzy logic for final decision-making


def fuzzy_logic(probabilities):
    categories = ['healthy', 'powdery', 'rust']
    # Define fuzzy variables
    health = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'health')
    powdery = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'powdery')
    rust = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rust')
    condition = ctrl.Consequent(np.arange(0, 101, 1), 'condition')

    # Define fuzzy membership functions
    health['low'] = fuzz.trimf(health.universe, [0, 0, 0.5])
    health['medium'] = fuzz.trimf(health.universe, [0, 0.5, 1])
    health['high'] = fuzz.trimf(health.universe, [0.5, 1, 1])

    powdery['low'] = fuzz.trimf(powdery.universe, [0, 0, 0.5])
    powdery['medium'] = fuzz.trimf(powdery.universe, [0, 0.5, 1])
    powdery['high'] = fuzz.trimf(powdery.universe, [0.5, 1, 1])

    rust['low'] = fuzz.trimf(rust.universe, [0, 0, 0.5])
    rust['medium'] = fuzz.trimf(rust.universe, [0, 0.5, 1])
    rust['high'] = fuzz.trimf(rust.universe, [0.5, 1, 1])

    condition['poor'] = fuzz.trimf(condition.universe, [0, 0, 50])
    condition['moderate'] = fuzz.trimf(condition.universe, [0, 50, 100])
    condition['good'] = fuzz.trimf(condition.universe, [50, 100, 100])

    # Define fuzzy rules
    rule1 = ctrl.Rule(health['high'] & powdery['low']
                      & rust['low'], condition['good'])
    rule2 = ctrl.Rule(health['medium'] & (
        powdery['medium'] | rust['medium']), condition['moderate'])
    rule3 = ctrl.Rule(health['low'] | powdery['high']
                      | rust['high'], condition['poor'])

    system = ctrl.ControlSystem([rule1, rule2, rule3])
    simulation = ctrl.ControlSystemSimulation(system)

    # Apply probabilities
    simulation.input["health"] = probabilities[0]
    simulation.input["powdery"] = probabilities[1]
    simulation.input["rust"] = probabilities[2]

    max = -1
    cla = ''
    for i in range(len(categories)):
        if max < probabilities[i]:
            max = probabilities[i]
            cla = categories[i]

    simulation.compute()

    return simulation.output["condition"], cla.capitalize()

# Generate text using Gemini (GenAI)


def generate_gemini_response(probabilities, condition):
    prompt = '''
    Based on the plant condition report with averaged probabilities and fuzzy condition percentage, explain:
    
    Why is the plant in its current state? Discuss key factors like rust, health, etc., using the probabilities.
    What caused this condition? Identify contributing factors such as diseases or care issues.
    What actions should be taken to improve or maintain health? Recommend specific care steps.
    Discuss more about plants current condition. also check probabilities for better result
    Also tell what happens if this continues to happens (only tell if it is not in good state)
    
    in 400 words max
    in points wise. dont write long paragraphs
    '''
    prompt = str(probabilities) + prompt
    genai.configure(api_key="AIzaSyCeXuyiC9UK6AB_ZRJV9MwuWP0CTeJj6i4")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


# Streamlit app layout
st.title("Welcome to Agroscan! - Plant health detection using deep learning")
st.markdown("Upload an image to detect the condition of the plant leaf using multiple models, fuzzy logic, and Gemini-generated insights.")

uploaded_file = st.file_uploader(
    "Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            prepared_image = prepare_image(image)
            avg_probabilities = predict_with_models(prepared_image)
            final_condition, outp = fuzzy_logic(avg_probabilities)
            genai_response = generate_gemini_response(
                avg_probabilities, final_condition)

        # Results
        st.subheader(f"Analysis Results: ")
        if outp == "Rust" or outp == "Powdery":
            annotated_text(
                "**Final Result**: ",
                annotation(f"{outp}", "-> Danger",
                           border="2px dashed red")
            )
        else:
            annotated_text(
                "**Final Result**: ",
                (f"{outp}", "-> Healthy")
            )
        # st.write(f"**Final Result**: {outp}")
        st.write(
            f"**Category Probabilities**: {dict(zip(categories, avg_probabilities))}")
        # st.write(f"**Final Condition Score**: {final_condition:.2f}")
        st.subheader("Recommendations")
        st.write(genai_response)
