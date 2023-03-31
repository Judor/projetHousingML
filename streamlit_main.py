from itertools import cycle
import matplotlib.pyplot as plt
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
                              ["Overview", "Model Analysis", "Prediction"])
for i in range(30):
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
# Check if 'edit_expander' exists in session_state, if not, set it to False

if "edit_expander" not in st.session_state:
    st.session_state.edit_expander = False

edit_button_container = st.sidebar.markdown(
    '<div id="edit-button-container"></div>',
    unsafe_allow_html=True
)
if navigation == "Overview":
    col1, coLX, col2, col3 = st.columns([3.5, 0.5, 1, 0.6])
    col1.write('')
    col1.title("House Price Prediction Model")
    col1.write('')
    st.write("""
        ###### This is a project for the HETIC's Data Science Master's Degree. We are Grégory Haton, Guillaume Lochon, Hugo Bacard, Anis Akeb & Sébastien Tadiello""")
    col2.write("")

# Load ML MODEL results
if navigation == "Model Analysis":
    st.write("### XGBoost Model Analysis")
    st.write(" ")
    model = 'xgboost'
    st.write(" ")
    st.write("")
    col1, col2, col3, col4, col5 = st.columns(5)
    model, features, mae, rmse, r2, mape, X_train, X_test, y_train, y_test = load_model(model)
    col1.metric(label="MAPE", value=str(round(mape, 2)) + " %",
                help="MAPE of the model", delta_color='off')
    col2.metric(label="R2", value=round(r2, 2),
                help="R2 of the model", delta_color='off')
    col3.metric(label="MAE", value=str(round(mae, 1)) + str(" $"),
                help="MAE of the model", delta_color='off')
    col4.metric(label="RMSE", value=str(round(rmse, 1)) + str(" $"),
                help="RMSE of the model", delta_color='off')
    col5.metric(label="Features", value=len(features),
                help="Number of features", delta_color='off')
    st.markdown('---')
    st.write("### Model's Residuals")
    st.write(" ")
    col1, col2 = st.columns(2)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    # Calculate residuals on training and validation sets
    residuals_train = y_train - y_train_pred
    residuals_valid = y_test - y_pred

    # Plot the model and residuals for the training and validation sets
    fig1, (ax1, ax3) = plt.subplots(ncols=2, figsize=(10, 5))
    fig2, (ax2, ax4) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.scatter(y_train, y_train_pred, s=20, alpha=0.5)
    ax1.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], '--r', linewidth=2)
    ax1.set_xlabel('Actual values')
    ax1.set_ylabel('Predicted values')
    ax1.set_title(f'Training set (R2={r2_score(y_train, y_train_pred):.2f})')

    ax2.scatter(y_test, y_pred, s=20, alpha=0.5)
    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)
    ax2.set_xlabel('Actual values')
    ax2.set_ylabel('Predicted values')
    ax2.set_title(f'Validation set (R2={r2_score(y_test, y_pred):.2f})')

    ax3.scatter(y_train_pred, residuals_train, s=20, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Training set')

    ax4.scatter(y_pred, residuals_valid, s=20, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted values')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Validation set')
    fig1.tight_layout()
    col1.pyplot(fig1)

    fig2.tight_layout()
    col2.pyplot(fig2)
    st.write(" ")
    st.markdown('---')
    st.write("### Model's SHAP values")
    st.write(" ")
    image_path = "assets/shap.png"
    col0, col1, col2 = st.columns([1, 4, 1])
    col1.image(image_path, caption="Shap Values")

if navigation == "Prediction":
    st.write("### Prediction based on the XGBoost model")
    col1, col2 = st.columns(2)
    model = 'xgboost'
    model, features, mae, rmse, r2, mape, X_train, X_test, y_train, y_test = load_model(model)
    input = {}
    col_to_write_cycle = cycle([col1, col2])
    for feature in features:
        col_to_write = next(col_to_write_cycle)
        type, model_element_list, full_element_list = featureTransformation(feature)
        if type == "categorical":
            input.update({str(feature): str(col_to_write.selectbox(getGoodName(feature), full_element_list))})
        elif type == "numerical":
            input.update({str(feature): str(col_to_write.number_input(getGoodName(feature), 2018))})
    if st.button("Predict"):
        input_df = input_to_dataframe(input)
        prediction = estimate_house_price(model, input_df)[0]
        st.write("### Estimated Price :" + str(round(prediction, 2)) + " $")

