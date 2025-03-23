import streamlit as st
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Art Critique Bot",
    page_icon="🎨",
    layout="wide"
)

# Simple CSS for clean look
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .upload-section {
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #cccccc;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 Art Critique Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your artwork", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, caption="Uploaded Artwork", use_container_width=True)
    
    with col2:
        st.subheader("Image Analysis")
        
        # Basic image information
        width, height = image.size
        st.write(f"📏 Dimensions: {width}x{height}")
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate average color
        avg_color = np.mean(img_array, axis=(0, 1))
        st.write("🎨 Average Colors:")
        st.write(f"R: {int(avg_color[0])}, G: {int(avg_color[1])}, B: {int(avg_color[2])}")
        
        # Basic composition analysis
        st.write("📊 Composition:")
        aspect_ratio = width / height
        if aspect_ratio > 1:
            st.write("• Landscape orientation")
        elif aspect_ratio < 1:
            st.write("• Portrait orientation")
        else:
            st.write("• Square format")
            
        # Brightness analysis
        brightness = np.mean(img_array)
        st.write("✨ Overall Brightness:")
        if brightness < 85:
            st.write("• Dark tones dominant")
        elif brightness < 170:
            st.write("• Mid tones dominant")
        else:
            st.write("• Light tones dominant")
            
        # Color variety
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        st.write("🌈 Color Variety:")
        if unique_colors < 1000:
            st.write("• Limited color palette")
        elif unique_colors < 10000:
            st.write("• Moderate color variety")
        else:
            st.write("• Rich color variety")
else:
    st.write("👆 Upload an image to get started!")
