# 🎾 Tennis Player Trajectory Prediction

This project presents an end-to-end system for predicting tennis player movement trajectories using computer vision and deep learning techniques. It extends a YOLOv5-based detection framework by integrating a transformer-based model to forecast player positions 2–3 seconds into the future. The system enables predictive analytics for real-time broadcasting, tactical coaching, and performance evaluation.

## 🚀 Key Features
- **Real-time object detection** using YOLOv5 for tennis player and ball tracking
- **Multi-object tracking** with DeepSORT for consistent player identification
- **Motion analysis** including velocity and acceleration vector extraction
- **Trajectory forecasting** using a Transformer encoder-decoder model
- **Visualization tools** for predicted vs. actual movement paths

## 📊 Performance Highlights
- **Short-term predictions** show strong accuracy with low displacement error
- **Long-term predictions** reveal challenges with higher error, motivating future improvements
- Evaluation metrics include ADE, FDE, and visual vector fields

## 📁 Dataset
- [Roboflow Tennis Ball Detection Dataset](https://app.roboflow.com/cathy-idvqa/tennis-ball-detection-5v8ol/1/export)
- Augmented with frame-extracted broadcast videos for increased realism

## 🔧 Future Directions
- Integrate pose estimation for joint-level motion modeling
- Use spatial context (court zones) and game-state awareness for better long-term predictions
- Compare physics-based vs. data-driven hybrid prediction models

## 📬 Contact
Developed by **Xujia Qin**  
Email: `qin.xuj@northeastern.edu`  
Khoury College of Computer Sciences, Northeastern University

