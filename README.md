The app as input receive the pdf and as output gives annotated pdf with json detections.

Create virtual environment
conda create your_env
conda activate your_env

install requiremnts.txt
pip install -r requirements.txt

IMPORTANT — Windows PDF rendering

PDF2Image requires Poppler installed.

Download Poppler:
https://github.com/oschwartz10612/poppler-windows/releases/

Extract → copy the bin/ path → add to system PATH.

run the app :
streamlit run app.py

✅ What detectors.py Contains
1. Portable paths

Auto-detects project directory.

Creates/uses models/ folder.

Loads one YOLO mode

✅ What app.py Contains
1. Streamlit UI

Page title + description

Left column: PDF uploader

Right column: confidence slider + DPI slider

2. Calls detect_in_pdf()

Processes entire PDF

Gets per-page annotated images + boxes JSON

3. Displays results




