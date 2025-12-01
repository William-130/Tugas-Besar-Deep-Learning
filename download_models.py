import os
import gdown
import streamlit as st

# Google Drive file IDs
MODEL_FILES = {
    'facenet_model_20251201_225633.pkl': {
        'url': 'https://drive.google.com/file/d/1ZqCUZsGfEZ68MRInbJc8MNBLSCX-kEkt/view?usp=sharing',
        'output': './models/facenet_model_20251201_225633.pkl'
    },
    'facenet_classifier_20251201_225633.pth': {
        'url': 'https://drive.google.com/file/d/1o4Sfj8PueD_4L74ebwwSIp0hfCtcZ3nF/view?usp=sharing',
        'output': './models/facenet_classifier_20251201_225633.pth'
    }
}

def download_model_from_gdrive(url, output_path):
    """Download file from Google Drive"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract file ID from Google Drive URL
    if 'drive.google.com' in url:
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            raise ValueError(f"Cannot extract file ID from URL: {url}")
        
        # Use gdown to download
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, output_path, quiet=False)
        return True
    else:
        raise ValueError("URL is not a Google Drive link")

def check_and_download_models():
    """Check if models exist, download if not"""
    for model_name, info in MODEL_FILES.items():
        output_path = info['output']
        
        if not os.path.exists(output_path):
            st.info(f"⬇️ Downloading {model_name}...")
            
            try:
                if info['url'] == 'REPLACE_WITH_GOOGLE_DRIVE_LINK':
                    st.error(f"❌ Please configure Google Drive link for {model_name} in download_models.py")
                    return False
                
                download_model_from_gdrive(info['url'], output_path)
                st.success(f"✅ Downloaded {model_name}")
            except Exception as e:
                st.error(f"❌ Failed to download {model_name}: {str(e)}")
                return False
        else:
            st.success(f"✅ {model_name} already exists")
    
    return True

if __name__ == "__main__":
    # Test download
    check_and_download_models()
