import streamlit as st
from functions import *
from PIL import Image

# TODO SIDEBAR
# Set the page configuration with a custom title, icon, and layout
st.set_page_config(page_title='Housing Project Groupe 11', page_icon=':bar_chart:', layout='wide')

# Insert the CSS styles into the page
css_content = read_css_file("app/style.css")
st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

# Load the inputs for the dashboard

# Load the light logo image
logo_image_light = Image.open("assets/hetic.png")

# Convert the light logo image to a base64 string
logo_base64_light = convert_image_to_base64(logo_image_light)

# Display the light logo in the sidebar
display_image_in_sidebar(logo_base64_light, width=300, margin_bottom=20)

st.sidebar.title("Navigation")
navigation = st.sidebar.radio("Select",
                              ["Features", "Models", "Prediction"])
for i in range(30):
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
# Check if 'edit_expander' exists in session_state, if not, set it to False

if "edit_expander" not in st.session_state:
    st.session_state.edit_expander = False

edit_button_container = st.sidebar.markdown(
    '<div id="edit-button-container"></div>',
    unsafe_allow_html=True
)

col1, coLX, col2, col3 = st.columns([3.5, 0.5, 1, 0.6])
col1.write('')
col1.title("House Price Prediction Model")
col1.write('')
st.write("""
    ###### This is a project for the HETIC's Data Science Master's Degree. We are Grégory Haton, Guillaume Lochon, Hugo Bacard, Anis Akeb & Sébastien Tadiello""")
col2.write("")

# Load ML MODEL results
if navigation == "Models":
    st.title("Models")
    genre = st.radio(
        "Please select a Model",
        ('XGBoost', 'Linear Regression', 'SVR', 'Random Forest'))
    if genre == 'XGBoost':
        model = 'xgboost'
    elif genre == 'Linear Regression':
        model = 'linear_regression'
    elif genre == 'SVR':
        model = 'svr'
    elif genre == 'Random Forest':
        model = 'random_forest'
    else:
        model = 'xgboost'
    st.write(genre)
    col1, col2, col3, col4, col5 = st.columns(5)
    # TODO load model function
    model, features, mae, rmse, r2, mape = load_model(model)
    col1.metric(label="MAPE", value=round(mape, 3),
                help="MAPE of the model", delta_color='off')
    col2.metric(label="R2", value=round(r2, 3),
                help="R2 of the model", delta_color='off')
    col3.metric(label="MAE", value=round(mae, 3),
                help="MAE of the model", delta_color='off')
    col4.metric(label="RMSE", value=round(rmse, 3),
                help="RMSE of the model", delta_color='off')
    col5.metric(label="Features", value=round(rmse, 3),
                help="RMSE of the model", delta_color='off')

if navigation == "Prediction":
    genre = st.radio(
        "Please select a Model",
        ('XGBoost', 'Linear Regression', 'SVR', 'Random Forest'))
    if genre == 'XGBoost':
        model = 'xgboost'
    elif genre == 'Linear Regression':
        model = 'linear_regression'
    elif genre == 'SVR':
        model = 'svr'
    elif genre == 'Random Forest':
        model = 'random_forest'
    else:
        model = 'xgboost'
    model, features, mae, rmse, r2, mape = load_model(model)
    input = {}
    for feature in features:
        type, model_element_list, full_element_list = featureTransformation(feature)
        if type == "categorical":
            input.update({feature: st.selectbox(feature, full_element_list)})
        elif type == "numerical":
            input.update({feature: st.number_input(feature)})

    prediction = estimate_house_price(features, model, input)
