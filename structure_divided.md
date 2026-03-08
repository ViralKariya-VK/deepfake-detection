deepfake-detection/
│
├── README.md                          [BOTH]
├── .gitignore                         [BOTH]
├── requirements.txt                   [BOTH]
├── config.yaml                        [BOTH]
│
├── data/
│   ├── __init__.py
│   ├── drive_loader.py                [YOU]
│   ├── dataset.py                     [YOU]
│   └── transforms.py                  [YOU]
│
├── preprocessing/
│   ├── __init__.py
│   ├── face_detector.py               [YOU ✅ DONE]
│   ├── face_aligner.py                [YOU ✅ DONE]
│   └── video_to_frames.py             [YOU ✅ DONE]
│
├── spatial_branch/
│   ├── __init__.py
│   ├── encoder.py                     [YOU ✅ DONE]
│   ├── extract_embeddings.py          [YOU ✅ DONE]
│   ├── anomaly.py                     [YOU ← NEXT]
│   └── autoencoder.py                 [YOU]
│
├── temporal_branch/
│   ├── __init__.py
│   ├── video_processor.py             [FRIEND]
│   ├── roi_extractor.py               [FRIEND]
│   ├── pos_algorithm.py               [FRIEND]
│   ├── signal_processing.py           [FRIEND]
│   └── vppg_scorer.py                 [FRIEND]
│
├── fusion/
│   ├── __init__.py
│   └── fusion.py                      [BOTH]
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     [YOU]
│   └── visualizer.py                  [YOU ✅ DONE]
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      [YOU]
│   ├── 02_face_detection_test.ipynb   [YOU]
│   ├── 03_dino_embeddings.ipynb       [YOU]
│   ├── 04_anomaly_detection.ipynb     [YOU]
│   ├── 05_autoencoder.ipynb           [YOU]
│   ├── 06_vppg_signal.ipynb           [FRIEND]
│   └── 07_fusion_experiments.ipynb    [BOTH]
│
└── app/
    ├── streamlit_app.py               [BOTH]
    └── utils.py                       [BOTH]