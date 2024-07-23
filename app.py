# import streamlit as st
# from fastai.vision.all import *
# import plotly.express as px
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# #title
# st.title('Transportni klassifikatsiya qiluvchi model')

# # rasmni joylash
# file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'svg'])
# if file:
#     st.image(file)

#     # PIL convert
#     img = PILImage.create(file)

#     # model
#     model = load_learner('transport_model.pkl')

#     # predict
#     pred, pred_id, probs = model.predict(img)
#     st.success(f"Bashorat: {pred}")
#     st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

#     #plotting
#     fig = px.bar(x=probs*100, y=model.dls.vocab)     
#     st.plotly_chart(fig)

import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

# Adjust path for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Page configuration
st.set_page_config(
    page_title="Transportni klassifikatsiya qiluvchi model",
    page_icon=":car:",
    layout="wide"
)

# Sidebar content with additional widgets
st.sidebar.title("Ayubxon Zaynobiddinov")
st.sidebar.markdown("[Telegram](https://t.me/Ayubkhan_22)")

# Ensure the image path is correct
image_path = "C:/Users/Victus/Pictures/Ayubkhanjpg.jpg"
try:
    st.sidebar.image(image_path, caption="Ayubxon Zaynobiddinov", use_column_width=True)
except FileNotFoundError:
    st.sidebar.error("Rasm topilmadi. Iltimos, rasm manzilini tekshiring.")

# Additional sidebar widgets for interaction
st.sidebar.markdown("### Foydalanuvchi ma'lumotlari")
name = st.sidebar.text_input("Ismingiz:")
feedback = st.sidebar.text_area("Fikr-mulohazangiz:")

# Main title with additional styling
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2em;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="main-title">Transportni klassifikatsiya qiluvchi model</div>', unsafe_allow_html=True)

# Image uploader
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'svg'])
if file:
    st.image(file, caption="Yuklangan rasm", use_column_width=True)

    # Convert to PIL Image
    img = PILImage.create(file)

    # Load model
    model = load_learner('transport_model.pkl')

    # Predict
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # Plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab, labels={'x': 'Ehtimollik (%)', 'y': 'Kategoriyalar'}, title="Kategoriyalar bo'yicha ehtimollik")
    fig.update_layout(xaxis_title="Ehtimollik (%)", yaxis_title="Kategoriyalar", title_x=0.5)
    st.plotly_chart(fig)

# Footer with some styling
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 0.9em;
    }
    </style>
    <div class="footer">
        <p>Developed by Ayubxon Zaynobiddinov | <a href="https://t.me/Ayubkhan_22" target="_blank">Telegram</a></p>
    </div>
    """,
    unsafe_allow_html=True
)


