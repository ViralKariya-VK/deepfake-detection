deepfake-detection/
│
├── README.md                        # Project overview, setup instructions
├── .gitignore                       # Ignore data, checkpoints, .env, __pycache__
├── requirements.txt                 # For local (M4 Mac)
├── requirements_colab.txt           # For Google Colab (sometimes differs)
├── config.yaml                      # All global configs (paths, hyperparams, flags)
│
├── data/                            # NO actual data here — only helpers
│   ├── __init__.py
│   ├── drive_loader.py              # Mounts Drive, returns dataset paths
│   ├── dataset.py                   # PyTorch Dataset classes
│   └── transforms.py                # Augmentations / preprocessing
│
├── preprocessing/
│   ├── __init__.py
│   ├── face_detector.py             # RetinaFace / MTCNN wrapper
│   ├── face_aligner.py              # Alignment logic
│   └── video_to_frames.py           # Video → frames → cropped faces
│
├── spatial_branch/
│   ├── __init__.py
│   ├── encoder.py                   # DINO ViT loader + embedding extractor
│   ├── anomaly.py                   # Isolation Forest / One-Class SVM
│   └── autoencoder.py               # Autoencoder for heatmap generation
│
├── temporal_branch/
│   ├── __init__.py
│   ├── rppg_extractor.py            # ROI extraction + RGB signal
│   ├── signal_processing.py         # Bandpass filter, FFT
│   └── vppg_scorer.py               # Final physiological score
│
├── fusion/
│   ├── __init__.py
│   └── fusion.py                    # Weighted / learned fusion of both branches
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # AUC-ROC, accuracy, frame-level eval
│   └── visualizer.py                # t-SNE, UMAP, heatmap overlays
│
├── notebooks/                       # Exploration — NOT production code
│   ├── 01_data_exploration.ipynb
│   ├── 02_face_detection_test.ipynb
│   ├── 03_dino_embeddings.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_vppg_signal.ipynb
│   └── 06_fusion_experiments.ipynb
│
├── colab/                           # Colab-specific notebooks (shared Drive)
│   ├── colab_preprocessing.ipynb    # Heavy preprocessing on Colab GPU
│   ├── colab_train_encoder.ipynb    # Fine-tuning on Colab
│   └── colab_eval.ipynb             # Full evaluation runs
│
├── checkpoints/                     # .gitignore this — store on Drive instead
│   └── .gitkeep
│
├── outputs/                         # .gitignore this — heatmaps, results, plots
│   └── .gitkeep
│
└── app/
    ├── streamlit_app.py             # Final demo dashboard
    └── utils.py                     # App-specific helpers