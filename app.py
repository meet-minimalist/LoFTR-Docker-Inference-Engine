##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-23 6:58:34 pm
# @copyright MIT License
#

from flask import Flask, render_template, request, send_from_directory
import os
import traceback
import time
from loftr_torch import InferenceEngine

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

infer_engine = InferenceEngine()


@app.route("/")
def index() -> str:
    """Funtion to render HTML Page.

    Returns:
        str: HTML contents
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload() -> str:
    """Function to provide image upload and model inference facility.

    Returns:
        str: Result message string.
    """
    try:
        if "image1" not in request.files or "image2" not in request.files:
            return "Both images not uploaded"

        image1 = request.files["image1"]
        image2 = request.files["image2"]

        if image1.filename == "" or image2.filename == "":
            return "Both images not selected"

        img_1_name = image1.filename
        img_2_name = image2.filename
        upload_path_1 = os.path.join(app.config["UPLOAD_FOLDER"], img_1_name)
        upload_path_2 = os.path.join(app.config["UPLOAD_FOLDER"], img_2_name)

        image1.save(upload_path_1)
        image2.save(upload_path_2)

        start = time.time()
        model_outputs, processed_path = infer_engine(
            upload_path_1, upload_path_2, app.config["PROCESSED_FOLDER"]
        )
        delta = time.time() - start
        
        download_url = f"/download/{os.path.basename(processed_path)}"
        return "Images uploaded and processed successfully! Model inference " \
            f"time: {delta:.2f} sec. Download link: <a href='{download_url}'>{download_url}</a>"

    except Exception as e:
        traceback.print_exc()
        return f"Error occurred: {str(e)}"


@app.route("/download/<filename>")
def download(filename: str):
    """Function to download the result image file from provided url.

    Args:
        filename (str): File name of the result image.
    """
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    port = 5000
    app.run(host="0.0.0.0", port=port)
