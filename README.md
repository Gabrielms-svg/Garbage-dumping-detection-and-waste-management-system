Illegal Garbage Dump Detection & Waste Management System (Dump Surveillance AI)
-------------------------------------------------------------------------------


Abstract
---------
Dump Surveillance AI is an AI-powered web application designed to detect, analyze, and manage illegal garbage dumping activities. 
The system leverages CCTV surveillance feeds and advanced computer vision techniques to automatically identify illegal dumping incidents and support effective waste management.
The application employs YOLO-based object detection to recognize dumping activities and extract vehicle number plates involved in the offense. Detected incidents are processed using clustering and analytical models to generate actionable insights. These insights are visualized on interactive maps using the Leaflet API, enabling authorities to identify and monitor illegal dumping hotspots efficiently.
In addition to surveillance-based detection, the platform allows citizens to report illegal dumping by uploading media along with time, location, and incident details. Users can also locate authorized legal dumping sites nearby, promoting responsible waste disposal. To enhance public awareness, the system includes an AI-powered chatbot that provides guidance on proper waste management practices.
Illegal waste dumping remains a significant environmental and public health challenge in India, primarily due to limited access to legal disposal facilities and inadequate offender accountability. Unregulated dumping leads to pollution, health hazards, and ecosystem degradation. Traditional manual monitoring methods are inefficient and resource-intensive. This project aims to automate surveillance and reporting, enabling government authorities to improve enforcement, ensure accountability, and promote cleaner and safer environments.


⚠️ Note: This project uses CUDA 11.8-enabled PyTorch for GPU-accelerated inference

requirements :
--------------

Programming Language
-------------------------------
Python: 3.11.2

Web Framework & Backend
-------------------------------
Django: 3.2.7

asgiref: 3.9.1

sqlparse: 0.5.3

pytz / tzdata: 2025.2

Database
-------------------------------
MySQL

mysqlclient: 2.2.7

Computer Vision & AI
-------------------------------
YOLO (Ultralytics): 8.3.237

OpenCV: 4.12.0.88

EasyOCR: 1.7.2

PyTorch: 2.7.1+cu118

TorchVision: 0.22.1+cu118

TorchAudio: 2.7.1+cu118

Machine Learning & Analytics
-------------------------------
NumPy: 2.2.6

Pandas: 2.3.3

Scikit-learn: 1.7.2

SciPy: 1.16.3

Joblib: 1.5.2

Visualization & Mapping
-------------------------------
Matplotlib: 3.10.8

Shapely: 2.1.2

Image & Signal Processing
------------------------------
Pillow: 11.3.0

Scikit-image: 0.26.0

ImageIO: 2.37.2

Tifffile: 2025.12.20

Utilities & Supporting Libraries
---------------------------------
Requests: 2.32.5

PyYAML: 6.0.3

Psutil: 7.1.3

NetworkX: 3.6.1

SymPy: 1.14.0
