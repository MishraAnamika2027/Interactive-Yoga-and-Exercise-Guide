# Real-time Yoga Pose Detection System

An intelligent real-time yoga pose detection and classification system built with Flask, OpenCV, and MediaPipe. This application uses computer vision and pose estimation to detect, analyze, and provide feedback on yoga poses through a web interface.

## 🚀 Tech Stack

### Backend
- **Python 3.x** - Core programming language
- **Flask** - Web framework for serving the application
- **OpenCV (cv2)** - Computer vision library for image/video processing
- **MediaPipe** - Google's ML solution for pose estimation and landmark detection
- **NumPy** - Numerical computing for angle calculations and data processing

### Frontend
- **HTML5** - Structure and markup
- **CSS3** - Styling and responsive design
- **JavaScript** - Client-side interactivity and AJAX requests

### Additional Libraries
- **Threading** - Concurrent camera access management
- **JSON** - Data serialization and storage
- **Pathlib** - Modern file system path handling
- **Platform/Subprocess** - System integration for Windows camera detection

### Data Processing
- **CSV** - Training data analysis output format
- **JSON** - Configuration and pose threshold storage

## 📋 Features

- **Real-time Pose Detection**: Live webcam feed with MediaPipe pose estimation
- **Pose Classification**: Automatic classification of yoga poses (T-Pose, Tree Pose, etc.)
- **Confidence Scoring**: Real-time confidence metrics for pose accuracy
- **Joint Angle Analysis**: Calculate and display key joint angles
- **Multi-Camera Support**: Detect and switch between multiple cameras (Windows)
- **Training Data Analysis**: Statistical analysis of pose training images
- **Web-based Interface**: Accessible through any modern web browser
- **RESTful API**: JSON endpoints for pose metrics and camera control

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Webcam or external camera
- Windows/Linux/macOS

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/selcia25/opencv-yoga-postures.git
   cd opencv-yoga-postures
   ```

2. **Install dependencies**:
   ```bash
   pip install Flask opencv-python mediapipe numpy
   ```

   Or install all dependencies at once:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
opencv-yoga-postures-main/
├── app.py                              # Main Flask application
├── analyze_training_data.py            # Training data analysis script
├── training_analysis_results.json      # Pose angle statistics
├── training_analysis_results.csv       # Pose analysis in CSV format
├── PROJECT_REPORT.md                   # Detailed project documentation
├── README.md                           # This file
├── Our_trained_data/                   # Training images directory
│   ├── T_Pose/                        # T-Pose training images
│   └── Tree_Pose/                     # Tree Pose training images
├── static/                             # Static assets
│   ├── home-style.css                 # Home page styles
│   ├── yoga-style.css                 # Yoga detection page styles
│   └── img/                           # Image assets
└── templates/                          # HTML templates
    ├── home.html                      # Landing page
    ├── index.html                     # Main application page
    └── yoga_try.html                  # Yoga practice interface
```

## 🎯 Usage

### Basic Usage

1. **Start the Application**: Run `python app.py`
2. **Access Web Interface**: Navigate to `http://localhost:5000`
3. **Grant Camera Access**: Allow browser to access your webcam
4. **Position Yourself**: Stand in front of the camera with your full body visible
5. **Perform Poses**: Execute yoga poses and receive real-time feedback

### Supported Poses

- **T-Pose**: Arms extended horizontally, legs straight
- **Tree Pose**: One leg bent with foot on opposite thigh, arms raised

### Camera Selection

On Windows systems, the application can detect multiple cameras:
- Click the camera selector in the web interface
- Choose your preferred camera from the dropdown
- The feed will automatically switch to the selected camera

### API Endpoints

- `GET /` - Home page
- `GET /poses` - Pose detection page
- `GET /yoga_try` - Yoga practice interface
- `GET /video_feed1` - Video stream endpoint
- `GET /pose_metrics` - JSON response with current pose data
- `GET /get_available_cameras` - List available cameras
- `POST /set_camera` - Switch to a specific camera

## 🔬 Training Data Analysis

To analyze your own training images:

```bash
python analyze_training_data.py --input Our_trained_data --output-prefix training_analysis_results
```

This generates:
- `training_analysis_results.json` - Statistical pose angle data
- `training_analysis_results.csv` - Tabular format for analysis

## ⚙️ Configuration

Key parameters in `app.py`:

```python
CONFIDENCE_THRESHOLD = 60.0              # Minimum confidence for pose classification
ENABLE_ANGLE_OVERLAY = True              # Show joint angles on video
ENABLE_CONFIDENCE_OVERLAY = True         # Show confidence scores
```

MediaPipe Pose settings:
```python
pose = mp_pose.Pose(
    static_image_mode=False,             # Optimized for video
    min_detection_confidence=0.5,        # Detection threshold
    model_complexity=1                   # 0=Lite, 1=Full, 2=Heavy
)
```

## 🐛 Troubleshooting

### Camera Not Working
- Ensure camera permissions are granted in browser
- Check if camera is being used by another application
- Try switching cameras using the camera selector

### Installation Issues
- Ensure Python 3.7+ is installed: `python --version`
- Try upgrading pip: `pip install --upgrade pip`
- On Windows, you may need Visual C++ redistributables for OpenCV

### Performance Issues
- Reduce MediaPipe model complexity in `app.py`
- Lower webcam resolution
- Close other resource-intensive applications

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add YourFeature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Submit a pull request.

## 📝 License

This project is open source and available under the MIT License.


## 🙏 Acknowledgments

- **Google MediaPipe** - Pose estimation technology
- **OpenCV** - Computer vision capabilities
- **Flask** - Web framework
- Yoga community for pose validation and testing

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Note**: This application is designed for educational and fitness guidance purposes. Always consult with a certified yoga instructor for proper pose techniques and safety.
