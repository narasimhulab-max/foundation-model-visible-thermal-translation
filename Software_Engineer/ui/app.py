import sys
from pathlib import Path

# -------------------------------------------------------------
# Add project root to Python path
# -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import APP_NAME
from src.logger import get_logger
from src.validation import (
    is_supported_image,
    is_supported_modality,
)
from src.input_handler import create_input


# -------------------------------------------------------------
# Application Logger
# -------------------------------------------------------------
logger = get_logger(__name__)


# -------------------------------------------------------------
# Main Application
# -------------------------------------------------------------
def main():

    # ---------------------------------------------------------
    # Page Configuration
    # ---------------------------------------------------------
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🌡️",
        layout="wide",
    )

    # ---------------------------------------------------------
    # Application Header
    # ---------------------------------------------------------
    st.title("🌡️ Cross-Spectral Image Translation")

    st.write(
        "Upload a thermal/infrared or night-vision image "
        "to generate a plausible visible-spectrum RGB image."
    )

    st.info(
        "The generated RGB image represents a plausible "
        "visible-spectrum interpretation of the input. "
        "The predicted colors are not guaranteed to represent "
        "the true physical colors of the scene."
    )

    st.divider()

    # ---------------------------------------------------------
    # Step 1: Select Input Modality
    # ---------------------------------------------------------
    st.subheader("1. Select Input Modality")

    input_modality = st.radio(
        "Choose the type of input image:",
        options=[
            "Thermal / Infrared",
            "Night Vision",
        ],
        horizontal=True,
    )

    # Convert UI label into internal modality name
    if input_modality == "Thermal / Infrared":
        modality = "thermal"
    else:
        modality = "night_vision"
    if not is_supported_modality(modality):
        st.error("Unsupported input modality.")
        logger.error("Unsupported input modality: %s", modality)
        return

    logger.info(
        "Selected input modality: %s",
        modality,
    )

    # ---------------------------------------------------------
    # Step 2: Upload Image
    # ---------------------------------------------------------
    st.subheader("2. Upload Input Image")

    uploaded_file = st.file_uploader(
        f"Choose a {input_modality.lower()} image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG",
    )

    # ---------------------------------------------------------
    # Process Uploaded Image
    # ---------------------------------------------------------
    if uploaded_file is not None:

        # -----------------------------------------------------
        # Validate Image Format
        # -----------------------------------------------------
        if not is_supported_image(uploaded_file.name):

            st.error(
                "Unsupported image format. "
                "Please upload a JPG, JPEG or PNG image."
            )

            logger.warning(
                "Unsupported file uploaded: %s",
                uploaded_file.name,
            )

            return

        # -----------------------------------------------------
        # Successful Upload
        # -----------------------------------------------------
        logger.info(
            "%s image uploaded successfully: %s",
            modality,
            uploaded_file.name,
        )
        try:
            input_data = create_input(
                image=uploaded_file,
                modality=modality,
                filename=uploaded_file.name,
            )
            logger.info(
                "Input successfully created | modality=%s | filename=%s",
                 input_data.modality,
                 input_data.filename,

            )
        except ValueError as error:
            st.error(str(error))
            logger.error("Input validation failed: %s", error)
            return
        st.success("Input successfully validated!")

        st.write("**Input details:**")

        st.write(
            {
                "filename": input_data.filename,
                "modality": input_data.modality,
            }
        )

        # -----------------------------------------------------
        # Step 3: Display Input Image
        # -----------------------------------------------------
        st.subheader("3. Input Image")

        st.image(
            uploaded_file,
            caption=f"{input_modality}: {uploaded_file.name}",
            width="stretch",
        )

        # Display information about the selected modality
        st.success(
            f"Input modality detected: **{input_modality}**"
        )

        st.divider()

        # -----------------------------------------------------
        # Step 4: Image Translation
        # -----------------------------------------------------
        st.subheader("4. Generate Visible RGB Image")

        generate_button = st.button(
            "Generate RGB Image",
            type="primary",
        )

        if generate_button:

            logger.info(
                "RGB generation requested | modality=%s | file=%s",
                modality,
                uploaded_file.name,
            )

            # -------------------------------------------------
            # Future ML Pipeline
            # -------------------------------------------------
            st.subheader("5. Generated Visible Image")

            st.info(
                "AI inference module will be connected here "
                "after the translation model is integrated."
            )

            # -------------------------------------------------
            # Current Pipeline Display
            # -------------------------------------------------
            st.caption(
                "Current pipeline:"
            )

            st.code(
                f"""
Input Modality
      ↓
{input_modality}
      ↓
Input Validation
      ↓
Preprocessing
      ↓
Multimodal Diffusion Model
      ↓
Visible RGB Image
""",
                language="text",
            )


# -------------------------------------------------------------
# Application Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()