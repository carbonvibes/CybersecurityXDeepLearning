import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.binary_to_image import binary_file_to_image, calculate_file_entropy
from utils.visualization import overlay_heatmap, GradCAM
from models.resnet import create_resnet_model
from utils.data_loader import get_val_transforms, get_class_names

# Page config
st.set_page_config(
    page_title="Malware Detective AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #f0f2f6;
        text-align: center;
    }
    .malware-detected {
        background-color: #ffebee;
        border-color: #ef5350;
        color: #c62828;
    }
    .safe-file {
        background-color: #e8f5e9;
        border-color: #66bb6a;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_config():
    """Load the trained model and configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find the best checkpoint
    checkpoint_dir = project_root / 'checkpoints'
    checkpoints = list(checkpoint_dir.glob('**/checkpoint_best.pth'))
    
    if not checkpoints:
        return None, None, None, None
    
    # Sort by modification time (newest first)
    latest_checkpoint = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        config = checkpoint['config']
        
        # Extract accuracy from checkpoint root if not in config
        if 'best_val_acc' in checkpoint:
            config['best_val_acc'] = checkpoint['best_val_acc']
        
        # Create model
        num_classes = config['num_classes']
        model = create_resnet_model(num_classes=num_classes, variant='resnet50', pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Get class names
        splits_dir = project_root / 'data' / 'splits'
        class_names = get_class_names(splits_dir)
        
        return model, config, class_names, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def process_file(uploaded_file):
    """Convert uploaded file to image."""
    # Save temporarily
    temp_path = Path("temp_upload")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Calculate entropy
    entropy = calculate_file_entropy(temp_path)
    
    # Convert to image
    try:
        img = binary_file_to_image(temp_path)
        return img, entropy, temp_path
    except Exception as e:
        st.error(f"Error converting file: {e}")
        return None, 0, temp_path

def main():
    # Sidebar
    st.sidebar.title("🛡️ Malware Detective")
    st.sidebar.info("Deep Learning based Malware Classification System")
    
    model, config, class_names, device = load_model_and_config()
    
    if model is None:
        st.error("No trained model found! Please train the model first.")
        return

    st.sidebar.success(f"Model Loaded: ResNet-50\nAccuracy: {config.get('best_val_acc', 0):.2f}%")
    
    mode = st.sidebar.radio("Mode", ["Live Analysis", "Model Insights", "About"])
    
    if mode == "Live Analysis":
        st.title("🔍 Live Malware Analysis")
        st.markdown("Upload a binary file (EXE, DLL) or a pre-converted image to classify it.")
        
        uploaded_file = st.file_uploader("Choose a file", type=['exe', 'dll', 'bin', 'png', 'jpg'])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("1. Visual Representation")
                with st.spinner("Processing file..."):
                    if uploaded_file.name.endswith(('.png', '.jpg', '.jpeg')):
                        img = Image.open(uploaded_file).convert('L')
                        entropy = 0 # Not calculated for images
                        temp_path = None
                    else:
                        img, entropy, temp_path = process_file(uploaded_file)
                    
                    st.image(img, caption="Binary-to-Image Conversion", width=300)
                    
                    if entropy > 0:
                        st.metric("File Entropy", f"{entropy:.2f}/8.0", 
                                 help="High entropy (>7.0) often indicates packed or encrypted malware.")
            
            with col2:
                st.subheader("2. AI Classification")
                
                if st.button("Analyze Malware Family"):
                    with st.spinner("Running Neural Network..."):
                        # Preprocess
                        transform = get_val_transforms(224)
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        
                        # Predict
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            probs = F.softmax(outputs, dim=1)
                            conf, pred = torch.max(probs, 1)
                            
                        pred_idx = pred.item()
                        confidence = conf.item()
                        family_name = class_names[pred_idx]
                        
                        # Display Result
                        st.markdown(f"""
                        <div class="prediction-box malware-detected">
                            <h2>🚨 {family_name}</h2>
                            <p>Confidence: <strong>{confidence:.2%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top 3
                        st.write("### Top Probabilities")
                        top3_prob, top3_idx = torch.topk(probs, 3)
                        for i in range(3):
                            p = top3_prob[0][i].item()
                            idx = top3_idx[0][i].item()
                            st.progress(p, text=f"{class_names[idx]}: {p:.1%}")

                        # Explainability
                        st.subheader("3. Explainability (Grad-CAM)")
                        gradcam = GradCAM(model)
                        heatmap = gradcam.generate(img_tensor, pred_idx)
                        
                        # Overlay
                        orig_np = np.array(img.resize((224, 224))) / 255.0
                        overlay = overlay_heatmap(orig_np, heatmap)
                        
                        st.image(overlay, caption="Heatmap: Red areas triggered the detection", use_column_width=True)
                        
            # Cleanup
            if temp_path and temp_path.exists():
                os.remove(temp_path)

    elif mode == "Model Insights":
        st.title("📊 Model Performance & Insights")
        
        # Load evaluation artifacts if they exist
        eval_dir = Path(config['checkpoint_dir']) / 'evaluation'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm_path = eval_dir / 'confusion_matrix_normalized.png'
            if cm_path.exists():
                st.image(str(cm_path), use_column_width=True)
            else:
                st.info("Confusion matrix not found. Run evaluation script first.")
                
        with col2:
            st.subheader("ROC Curves")
            roc_path = eval_dir / 'roc_curves.png'
            if roc_path.exists():
                st.image(str(roc_path), use_column_width=True)
            else:
                st.info("ROC curves not found.")
                
        st.subheader("Training History")
        # You could load training_history.json here and plot with plotly
        st.info("Training logs available in TensorBoard.")

    elif mode == "About":
        st.title("ℹ️ About This Project")
        st.markdown("""
        ### Malware Detection using Deep Learning
        
        This project uses **Computer Vision** techniques to detect malware families.
        
        #### How it works:
        1. **Binary to Image**: The raw bytes of a malware file are read as pixel values (0-255).
        2. **Reshaping**: The byte stream is reshaped into a 2D image.
        3. **CNN Analysis**: A ResNet-50 Convolutional Neural Network analyzes the visual patterns.
        4. **Classification**: The model predicts the specific malware family.
        
        #### Why this works:
        Malware authors often reuse code. This reused code creates consistent "visual textures" 
        in the binary image, which CNNs are incredibly good at recognizing.
        """)

if __name__ == "__main__":
    main()
