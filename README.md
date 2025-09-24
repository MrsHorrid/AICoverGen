# AICoverGen
An autonomous pipeline to create covers with any RVC v2 trained AI voice from YouTube videos or a local audio file. For developers who may want to add a singing functionality into their AI assistant/chatbot/vtuber, or for people who want to hear their favourite characters sing their favourite song.

<img width="1574" height="740" alt="image" src="https://github.com/user-attachments/assets/931189d8-e2e2-4240-84d6-52d7a13ac7f8" />

# AICoverGen Enhanced
**AI-Powered Voice Cover Generation with Advanced Audio Enhancement**

AICoverGen Enhanced is a powerful tool for creating AI voice covers with professional-grade audio enhancement features. This enhanced version includes advanced AI audio processing, EQ controls, dynamic range compression, and much more!

---

## ✨ New Features

### Advanced Audio Enhancement
- **AI Noise Reduction** – Remove background noise and artifacts  
- **Professional EQ** – 5 EQ types: Balanced, Vocal Boost, Bass Boost, Treble Boost, Flat  
- **Dynamic Range Compression** – Improve loudness and consistency  
- **Harmonic Enhancement** – Add richness and warmth to vocals  
- **Stereo Widening** – Enhance spatial imaging for stereo tracks  
- **Reverb Control** – Add depth and professional polish  
- **Gain Control** – Fine-tune volume (-20 to +20 dB)  

### Enhancement Types
- **Full** – Balanced enhancement with all features  
- **Light** – Subtle improvements for natural sound  
- **Aggressive** – Maximum enhancement for impact  
- **Custom** – Use your specific settings  

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, Linux, or macOS  
- **Python:** 3.9+ (3.10+ recommended)  
- **RAM:** 8GB minimum, 16GB recommended  
- **Storage:** 10GB free space  
- **GPU:** NVIDIA GPU with CUDA support (recommended)  

### Recommended Setup
- **OS:** Windows 11 or Ubuntu 20.04+  
- **Python:** 3.10 or 3.11  
- **RAM:** 16GB or more  
- **GPU:** NVIDIA RTX 3060 or better  
- **CUDA:** 11.8 or 12.0+  
- **cuDNN:** 8.6 or 9.0+  

---

## ⚙️ Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/SociallyIneptWeeb/AICoverGen.git
cd AICoverGen
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv AICoverGen
AICoverGen\Scripts\activate

# Linux/macOS
python3 -m venv AICoverGen
source AICoverGen/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Automatic Installation (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
# Core dependencies
pip install gradio==3.50.2 librosa==0.9.1 numpy==1.23.5 scipy==1.11.1 soundfile==0.12.1
pip install pedalboard==0.7.7 pydub==0.25.1 fairseq==0.12.2 faiss-cpu==1.7.3 pyworld==0.3.4
pip install praat-parselmouth>=0.4.2 ffmpeg-python>=0.2.0 tqdm==4.65.0 yt-dlp>=2025.9.23 sox==1.4.1

# AI Audio Enhancement dependencies
pip install noisereduce==3.0.3 scikit-learn==1.6.1

# PyTorch with CUDA support
pip install torch==2.0.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html
pip install torchcrepe==0.0.20

# ONNX Runtime with CUDA support
pip install onnxruntime-gpu==1.18.0
```

### Step 4: Download Models
```bash
python src/download_models.py
```

### Step 5: Verify Installation
```bash
python src/audio_enhancer.py
```

---

## 🚀 Usage

### Quick Start
```bash
python src/webui.py
```
Then open your browser and go to: [http://127.0.0.1:7860](http://127.0.0.1:7860)

1. Upload a song (YouTube URL or audio file)  
2. Select a voice model from the dropdown  
3. Configure audio enhancement:  
   - Expand "AI Audio Enhancement" section  
   - Choose enhancement type (Full/Light/Aggressive/Custom)  
   - Adjust EQ type (Balanced/Vocal Boost/Bass Boost/Treble Boost/Flat)  
   - Set noise reduction strength (0–100%)  
   - Adjust gain (-20 to +20 dB)  
   - Set compression ratio (1–10)  
   - Add reverb amount (0–100%)  
4. Click **Generate** and enjoy your enhanced AI cover!  

---

## 🛠️ Troubleshooting

### CUDA Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Check ONNX Runtime CUDA
python -c "import onnxruntime as ort; print('CUDA' in ort.get_available_providers())"
```

### Audio Enhancement Errors
```bash
# Test audio enhancer
python src/audio_enhancer.py

# Check dependencies
pip list | grep -E "(noisereduce|scikit-learn|pedalboard)"
```

### Memory Issues
- Reduce batch size in settings  
- Use CPU-only mode for ONNX Runtime  
- Close other applications to free RAM  

---

## 📂 Project Structure
```
AICoverGen_Enhanced/
├── src/
│   ├── webui.py             # Main web interface
│   ├── main.py              # Core pipeline with audio enhancement
│   ├── audio_enhancer.py    # AI audio enhancement module
│   ├── rvc.py               # RVC voice conversion
│   ├── mdx.py               # Audio separation
├── rvc_models/              # Voice models
├── mdxnet_models/           # Audio separation models
├── song_output/             # Generated covers
├── requirements.txt         # Dependencies
├── README_Enhanced.md       # This file
```

---

## 🎚️ Audio Enhancement Features

### AI Noise Reduction
- Uses ML to identify and remove background noise  
- Preserves vocal clarity while eliminating artifacts  
- Adjustable strength (0–100%)  

### Professional EQ
- **Balanced**: Gentle mid boost for clarity  
- **Vocal Boost**: Emphasizes 800–3000 Hz range  
- **Bass Boost**: Enhances 60–250 Hz  
- **Treble Boost**: Brightens 4–16 kHz  
- **Flat**: Minimal processing with high-pass filter  

### Dynamic Range Compression
- Improves loudness consistency  
- Reduces dynamic range for streaming  
- Configurable ratio (1–10)  

### Harmonic Enhancement
- Adds warmth and richness  
- Uses soft saturation for natural harmonics  

### Stereo Widening
- Improves spatial imaging  
- Enhances left-right separation  
- Creates immersive experience  

### Reverb Control
- Adds subtle depth and space  
- Professional room simulation  
- Configurable wet/dry mix  

---

## 🤝 Contributing
We welcome contributions! Please see our **Contributing Guidelines** for details.

---

## ⚖️ License
This project is licensed under the **MIT License** – see the LICENSE file for details.

---

## 🙏 Acknowledgments
- Original AICoverGen by **SociallyIneptWeeb**  
- RVC (Retrieval-based Voice Conversion) framework  
- MDXNet for audio separation  
- All the amazing open-source audio processing libraries  

