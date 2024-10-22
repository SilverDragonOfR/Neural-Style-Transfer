import io
import time
import streamlit as st
import os
from PIL import Image

from slow_neural_style_transfer import slow_neural_style_transfer
from fast_neural_style_transfer import fast_neural_style_transfer

def on_load():
    st.set_page_config(
        page_title="Artistic Style Transfer",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("✨ Artistic Style Transfer")

on_load()

with st.sidebar:
    st.write('# Settings')
    st.write('---')
    with st.expander("Slow NST"):
        nst_epochs = st.slider('Epochs (NST) :', min_value=500, max_value=3000, value=2000, step=100)
        nst_learning_rate = st.slider('Learning Rate (NST) :', min_value=0.0005, max_value=0.010, value=0.001, step=0.0005, format="%.4f")
        nst_alpha = st.slider('Alpha (NST) :', min_value=1, max_value=20, value=1, step=1)
        nst_beta =st.slider('Beta (NST) :', min_value=0.005, max_value=0.10, value=0.01, step=0.005, format="%.3f")
    with st.expander("Fast NST"):
        fnst_checkpoint = st.slider('Checkpoint (FNST)', min_value=500, max_value=7000, value=1500, step=500)

slow_nst_tab, fast_nst_tab = st.tabs(["Slow NST", "Fast NST"])

with slow_nst_tab:
    st.header("Slow Neural Style Transfer")
    st.caption("This method uses a deep neural network to transfer the artistic style from one image to another, typically producing high-quality results but at a slower speed.")

    container1 = st.container()
    col1, col2 = st.columns([8, 1], gap='medium')
    with container1:
        st.write("#### Upload Content Image :")
        with col1:
            nst_content_image = st.file_uploader("Choose a content image - Slow", type=["jpg", "jpeg", "png"])
        with col2:
            if nst_content_image:
                st.image(nst_content_image)
                
    container2 = st.container()
    col3, col4 = st.columns([8, 1], gap='medium')
    with container2:
        st.write("#### Upload Style Image :")
        with col3:
            nst_style_image = st.file_uploader("Choose a style image", type=["jpg", "jpeg", "png"])
        with col4:
            if nst_style_image:
                st.image(nst_style_image)
        
    st.write('---')

    is_nst_generate_btn_disabled = (nst_content_image == None) or (nst_style_image == None) or (nst_epochs == None) or (nst_learning_rate == None) or (nst_alpha == None) or (nst_beta == None)
    nst_generate_btn = st.button("Generate Slow", type='primary', disabled=is_nst_generate_btn_disabled)
        
    if nst_generate_btn:
        
        def progress(p, time_remaining, time_elapsed, loss):
            with result_holder.container(border=True):
                result_progress.progress(p, f'Progress : {p} %')
                st.write("#### Output :")
                st.write(time_remaining)
                st.write(time_elapsed)
                st.write(loss)

        @st.fragment
        def download_image(output_img):
            output_image_name = output_img.filename
            output_img_buffer = io.BytesIO()
            output_img.save(output_img_buffer, format="PNG")
            output_img_buffer.seek(0)
            st.download_button(label=f"Download {output_image_name}", data=output_img_buffer, icon="🔥", file_name=output_image_name, mime="image/png")
        
        result_holder = st.empty()
        result_progress = st.empty()
        
        output_img = slow_neural_style_transfer(nst_content_image, nst_style_image, epochs=nst_epochs, learning_rate=nst_learning_rate, alpha=nst_alpha, beta=nst_beta, callback=progress)
        
        with result_holder.container(border=True):
            result_progress.empty()
            st.write("#### Output :")
            st.image(output_img)
            download_image(output_img)

with fast_nst_tab:
    st.header("Fast Neural Style Transfer")
    st.caption("This method applies pre-trained neural networks to transfer artistic styles in real-time, producing results much faster but with slightly lower quality.")

    container3 = st.container()
    col5, col6 = st.columns([8, 1], gap='medium')
    with container3:
        st.write("#### Upload Content Image :")
        with col5:
            fnst_content_image = st.file_uploader("Choose a content image - Fast", type=["jpg", "jpeg", "png"])
        with col6:
            if fnst_content_image:
                st.image(fnst_content_image)
                
    container4 = st.container()
    col7, col8 = st.columns([8, 1], gap='medium')
    with container4:
        st.write("#### Choose a style :")
        with col7:
            style_options = [file.split('.')[0] for file in os.listdir(f"./fnst_pretrained_styles/styles/")]
            fnst_selected_style = st.selectbox("Image", style_options, index=None, placeholder="< Pretrained image >")
        with col8:
            if fnst_selected_style:
                st.image(f"./fnst_pretrained_styles/styles/{fnst_selected_style}.png")
                
    st.write('---')
    
    is_fnst_generate_btn_disabled = (fnst_content_image == None) or (fnst_selected_style == None) or (fnst_checkpoint == None)
    fnst_generate_btn = st.button("Generate Fast", type='primary', disabled=is_fnst_generate_btn_disabled)
    
    if fnst_generate_btn:

        @st.fragment
        def download_image(output_img):
            output_image_name = os.path.basename(output_img.filename)
            output_img_buffer = io.BytesIO()
            output_img.save(output_img_buffer, format="PNG")
            output_img_buffer.seek(0)
            st.download_button(label=f"Download {output_image_name}", data=output_img_buffer, icon="🔥", file_name=output_image_name, mime="image/png")
        
        result_holder = st.empty()
        result_holder.empty()
        
        with st.spinner('Wait for it...'):
            output_img = fast_neural_style_transfer(fnst_content_image, fnst_selected_style, fnst_checkpoint)
            
        with result_holder.container(border=True):
            st.write("#### Output :")
            st.image(output_img)
            download_image(output_img)
            