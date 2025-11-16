project/
│
├── app.py
├── detectors.py
├── README.md
│
├── models/
│   └── seals.pt        # your custom YOLOv8 weights
    └── signatures.pt  
│
└── requirements.txt

Create virtual environment
conda create your_env
conda activate your_env

install requiremnts.txt
pip install -r requirements.txt

MPORTANT — Windows PDF rendering

PDF2Image requires Poppler installed.

Download Poppler:
https://github.com/oschwartz10612/poppler-windows/releases/

Extract → copy the bin/ path → add to system PATH.

streamlit run app.py


