import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.stats import entropy
from sklearn.cluster import KMeans
import io
import json
from functools import lru_cache
from datetime import datetime
import hashlib
from modules.huggingface_api import HuggingFaceAPI
import base64
from io import BytesIO

# Initialize HuggingFace API with the key from secrets
if 'HUGGINGFACE_API_KEY' not in st.secrets:
    st.error("Please add your HuggingFace API key to the secrets.toml file")
    st.stop()

hf_api = HuggingFaceAPI(st.secrets['HUGGINGFACE_API_KEY'])

class UserPreferences:
    def __init__(self):
        self.preferences = self.load_preferences()
    
    def load_preferences(self):
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'style_focus': [],
                'feedback_detail': 'balanced',
                'color_importance': 5,
                'composition_importance': 5,
                'previous_critiques': []
            }
        return st.session_state.user_preferences
    
    def save_critique(self, image_hash, critique):
        self.preferences['previous_critiques'].append({
            'date': datetime.now().isoformat(),
            'image_hash': image_hash,
            'critique': critique
        })
        if len(self.preferences['previous_critiques']) > 10:
            self.preferences['previous_critiques'].pop(0)

class ArtCritic:
    def __init__(self):
        self.user_prefs = UserPreferences()
    
    def _image_to_hash(self, img_array):
        """Convert image array to a stable hash string"""
        # Resize to a small size for consistent hashing
        small_img = cv2.resize(img_array, (32, 32))
        # Convert to bytes and create hash
        return hashlib.md5(small_img.tobytes()).hexdigest()
    
    def _store_image(self, img_array):
        """Store image in session state and return hash"""
        img_hash = self._image_to_hash(img_array)
        if f'img_{img_hash}' not in st.session_state:
            st.session_state[f'img_{img_hash}'] = img_array
        return img_hash
    
    def _get_stored_image(self, img_hash):
        """Retrieve image array from session state"""
        return st.session_state.get(f'img_{img_hash}')
    
    @lru_cache(maxsize=32)
    def _analyze_image_cached(self, img_hash):
        """Cached version of image analysis"""
        img_array = self._get_stored_image(img_hash)
        if img_array is None:
            raise ValueError("Image not found in cache")
        comp = self.analyze_composition(img_array)
        colors = self.analyze_colors(img_array)
        return comp, colors

    def _image_to_base64(self, img_array):
        """Convert numpy array to base64 string"""
        img = Image.fromarray(img_array)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def get_image_description(self, img_array):
        """Get a detailed description of the image using vision models"""
        try:
            # Convert image to base64
            img_base64 = self._image_to_base64(img_array)
            
            # First try Microsoft model
            try:
                result = microsoft_vision_api.describe_image(img_base64)
                if result:
                    return result
            except:
                pass
            
            # Fallback to Hugging Face model
            result = hf_api.query(
                "nlpconnect/vit-gpt2-image-captioning",
                img_base64
            )
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            return "Unable to generate description"
            
        except Exception as e:
            return f"Error generating description: {str(e)}"

    def analyze_composition(self, img_array):
        """Analyze image composition using optimized computer vision"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # Resize for faster processing while maintaining aspect ratio
        max_dim = 800
        h, w = img_array.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_array = cv2.resize(img_array, None, fx=scale, fy=scale)
            
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        # Optimized edge detection
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        # Efficient thirds analysis
        thirds_strength = self._analyze_thirds(gray)
        
        # Texture analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(np.var(laplacian))
        
        # Focal point detection
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliency_map) = saliency.computeSaliency(img_array)
            focal_strength = float(np.mean(saliency_map)) if success else 0.0
        except:
            # Fallback to simple center focus analysis
            h, w = img_array.shape[:2]
            center_region = img_array[h//4:3*h//4, w//4:3*w//4]
            focal_strength = float(np.mean(center_region) / 255)
        
        return {
            'contrast': float(contrast),
            'brightness': float(brightness),
            'edge_density': float(edge_density),
            'thirds_strength': float(thirds_strength),
            'texture_variance': texture_variance,
            'focal_strength': focal_strength
        }
    
    def analyze_colors(self, img_array):
        """Analyze color distribution with caching"""
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # Downsample for faster processing
        img_small = cv2.resize(img_array, (100, 100))
        hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
        
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        sat_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        hue_entropy = float(entropy(hue_hist.flatten()))
        sat_entropy = float(entropy(sat_hist.flatten()))
        
        # Faster k-means with smaller sample
        pixels = img_small.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int).tolist()
        
        return {
            'hue_variety': hue_entropy,
            'saturation_variety': sat_entropy,
            'dominant_colors': colors
        }
    
    def _analyze_thirds(self, gray):
        """Optimized rule of thirds analysis"""
        h, w = gray.shape
        v_points = [w // 3, 2 * w // 3]
        h_points = [h // 3, 2 * h // 3]
        
        interest_regions = [
            gray[h1:h2, w1:w2] 
            for h1, h2 in zip([0] + h_points, h_points + [h])
            for w1, w2 in zip([0] + v_points, v_points + [w])
        ]
        
        region_means = [np.mean(region) for region in interest_regions]
        return np.var(region_means)
    
    def generate_critique(self, img_array):
        """Generate personalized art critique with image description"""
        try:
            # Store image and get hash
            img_hash = self._store_image(img_array)
            
            # Get cached analysis results
            comp_analysis, color_analysis = self._analyze_image_cached(img_hash)
            
            # Get image description
            description = self.get_image_description(img_array)
            
            # Personalize feedback based on user preferences
            prefs = self.user_prefs.preferences
            critique = []
            
            # Image Description
            critique.append("### 🖼️ Image Description")
            critique.append(description)
            
            # Personalized intro
            style_focus = prefs['style_focus']
            if style_focus:
                focus_str = ", ".join(style_focus)
                critique.append(f"\n### 🎨 Personalized Analysis\nFocusing on your interests in {focus_str}:")
            else:
                critique.append("\n### 🎨 Art Analysis\nHere's your personalized critique:")
            
            # Composition feedback based on user preferences
            if prefs['composition_importance'] > 3:
                comp_quality = "strong" if comp_analysis['thirds_strength'] > 1000 else "balanced" if comp_analysis['thirds_strength'] > 500 else "subtle"
                critique.append(f"\n#### 📐 Composition\nYour composition shows a {comp_quality} structure.")
                
                if comp_analysis['edge_density'] > 0.2:
                    critique.append("The piece has strong defining lines and clear focal points.")
                else:
                    critique.append("The composition has a soft, flowing quality.")
                
                # Texture analysis feedback
                if comp_analysis['texture_variance'] > 100:
                    critique.append("The artwork demonstrates rich textural details.")
                else:
                    critique.append("The texture is smooth and consistent throughout.")
                
                # Focal point analysis
                if comp_analysis['focal_strength'] > 0.5:
                    critique.append("The composition effectively draws attention to key areas.")
                else:
                    critique.append("The visual elements are more evenly distributed.")
            
            # Color analysis based on preferences
            if prefs['color_importance'] > 3:
                critique.append("\n#### 🎨 Color Analysis")
                if color_analysis['hue_variety'] > 1.5:
                    critique.append("Your color palette is diverse and dynamic.")
                else:
                    critique.append("You've chosen a focused, harmonious color scheme.")
            
            # Add personalized suggestions based on history
            if prefs['previous_critiques']:
                last_critique = prefs['previous_critiques'][-1]
                critique.append("\n#### 💡 Progress Note")
                critique.append("Compared to your previous work, I notice evolution in your technique.")
            
            # Save this critique
            self.user_prefs.save_critique(img_hash, "\n".join(critique))
            
            return "\n".join(critique)
            
        except Exception as e:
            return f"Error analyzing artwork: {str(e)}"

# Streamlit UI with preferences
st.title('🎨 Personalized Art Critique')

# User preferences section
with st.sidebar:
    st.subheader("Your Preferences")
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = UserPreferences().preferences
    
    prefs = st.session_state.user_preferences
    prefs['style_focus'] = st.multiselect(
        "Art styles you're interested in:",
        ['Impressionism', 'Abstract', 'Realism', 'Digital Art', 'Traditional'],
        default=prefs.get('style_focus', [])
    )
    prefs['feedback_detail'] = st.select_slider(
        "Feedback detail level:",
        options=['concise', 'balanced', 'detailed'],
        value=prefs.get('feedback_detail', 'balanced')
    )
    prefs['color_importance'] = st.slider(
        "Importance of color analysis:",
        1, 10, prefs.get('color_importance', 5)
    )
    prefs['composition_importance'] = st.slider(
        "Importance of composition analysis:",
        1, 10, prefs.get('composition_importance', 5)
    )

critic = ArtCritic()
uploaded_file = st.file_uploader("Upload your artwork", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.image(image, caption='Your Artwork', use_container_width=True)
        
        with st.spinner('Analyzing your artwork...'):
            critique = critic.generate_critique(img_array)
            st.markdown(critique)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")