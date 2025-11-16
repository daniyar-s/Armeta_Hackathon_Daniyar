# app.py

import streamlit as st
from detectors import detect_in_pdf
from io import BytesIO
from PIL import Image
import os

st.set_page_config(
    page_title="PDF QR / Seal / Signature Detector",
    layout="wide",
)

st.title("📄 QR / Seal / Signature Detector")
st.markdown(
    """
Upload a **PDF document**.  
The app will detect:

- 🟢 **QR codes**  
- 🔴 **Seals**  
- 🔵 **Signatures and Seal**  

and draw bounding boxes on each page.
"""
)

# -------------------------
# Helper: Build PDF
# -------------------------
def make_annotated_pdf(pages):
    """Create a multi-page PDF from a list of PIL images."""
    if not pages:
        return None
    rgb_pages = [p.convert("RGB") for p in pages]
    buf = BytesIO()
    first_page, *rest = rgb_pages
    first_page.save(buf, format="PDF", save_all=True, append_images=rest)
    buf.seek(0)
    return buf.getvalue()


col_left, col_right = st.columns([2.5, 1])

with col_right:
    conf_yolo = st.slider(
        "YOLO confidence threshold",
        0.1, 0.9, 0.35, 0.05
    )
    dpi = st.slider(
        "PDF render DPI",
        100, 300, 200, 25,
        help="Higher DPI → sharper images but slower"
    )

uploaded_pdf = col_left.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    # Extract filename for outputs
    original_name = uploaded_pdf.name
    base_name, _ = os.path.splitext(original_name)
    annotated_filename = f"{base_name}_annotated.pdf"
    detections_filename = f"{base_name}_detections.json"

    with st.spinner("Running detection models..."):
        results_per_page, boxes_json = detect_in_pdf(
            uploaded_pdf,
            conf_yolo=conf_yolo,
            dpi=dpi
        )

    st.success("Processing complete!")

    annotated_pages = []

    for page in results_per_page:
        st.subheader(f"Page {page['page_index']}")
        img = page["image_with_boxes"]
        annotated_pages.append(img)

        st.image(img, use_column_width=True)

        if page["all_boxes"]:
            st.caption("Detections:")
            st.dataframe(page["all_boxes"], use_container_width=True)
        else:
            st.info("No detections on this page.")

    st.subheader("📥 Export Results")

    # JSON export
    st.download_button(
        label="Download detections (JSON)",
        data=boxes_json,
        file_name=detections_filename,
        mime="application/json",
    )

    # Annotated PDF export
    pdf_bytes = make_annotated_pdf(annotated_pages)
    if pdf_bytes:
        st.download_button(
            label="Download annotated PDF",
            data=pdf_bytes,
            file_name=annotated_filename,
            mime="application/pdf",
        )
else:
    st.info("⬆️ Upload a PDF to begin.")
