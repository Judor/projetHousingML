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
inputs = load_inputs()

# Load the light logo image
logo_image_light = Image.open("assets/hetic.png")

# Convert the light logo image to a base64 string
logo_base64_light = convert_image_to_base64(logo_image_light)

# Display the light logo in the sidebar
display_image_in_sidebar(logo_base64_light, width=300, margin_bottom=20)

st.sidebar.title("Choose your model")
navigation = st.sidebar.radio("Model", ["XGBoost", "Random Forest", "SVR", "Linea Regression"])
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
if navigation == "XGBoost":
    st.write('')
    st.write('')
    col1, col2, col3, col4 = st.columns(4)
    mape, r2, mae, rmse = 0.1, 0.98, 0.2, 0.3
    col1.metric(label="MAPE", value=round(mape, 3),
                help="MAPE of the model", delta_color='off')
    col2.metric(label="R2", value=round(r2, 3),
                help="R2 of the model", delta_color='off')
    col3.metric(label="MAE", value=round(mae, 3),
                help="MAE of the model", delta_color='off')
    col4.metric(label="RMSE", value=round(rmse, 3),
                help="RMSE of the model", delta_color='off')
