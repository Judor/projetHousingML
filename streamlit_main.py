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

st.sidebar.title("Navigation")
navigation = st.sidebar.radio("Go to", ["Overview", "Campaign Performance Analysis", "Domains"])
for i in range(30):
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
# Check if 'edit_expander' exists in session_state, if not, set it to False

if "edit_expander" not in st.session_state:
    st.session_state.edit_expander = False

edit_button_container = st.sidebar.markdown(
    '<div id="edit-button-container"></div>',
    unsafe_allow_html=True
)

# TODO MAIN
