
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify
)

import json
import os
import base64
import logging
import random
import string
import qrcode
from datetime import datetime, timedelta

import cv2
import numpy as np


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

# Absolute path of the DigiYatra project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialise Flask
app = Flask(__name__)

# Temporary secret key for local development.
# Before public deployment, configure DIGIYATRA_SECRET_KEY
# as an environment variable with a strong random value.
secret_key = os.environ.get("DIGIYATRA_SECRET_KEY")

if not secret_key:
    raise RuntimeError(
        "DIGIYATRA_SECRET_KEY environment variable is not set."
    )

app.secret_key = secret_key

# Centralised directories
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TEMP_VISITORS_DIR = os.path.join(BASE_DIR, "temp_visitors")
QR_CODES_DIR = os.path.join(
    BASE_DIR,
    "static",
    "qrcodes"
)

# Centralised file paths

CASCADE_FILE = os.path.join(
    DATASET_DIR,
    "haarcascade_frontalface_default.xml"
)

# Maximum accepted camera image size: 5 MB
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Experimental LBPH threshold.
# This is a demonstration value, not a security standard.
MATCH_THRESHOLD = 70

# Limit incoming HTTP requests.
# Allows room for Base64 encoding overhead.
app.config["MAX_CONTENT_LENGTH"] = 7 * 1024 * 1024

# Session cookie security settings
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("SESSION_COOKIE_SECURE", "False").lower() == "true"
)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

@app.after_request
def prevent_page_caching(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Local development logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 2. JSON FILE HANDLING
# ============================================================

def load_json_file(file_path):

    """Load a JSON file using UTF-8 encoding."""

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 3. CAMERA IMAGE VALIDATION
# ============================================================

def decode_camera_image(image_data):

    """Validate and decode a browser camera image."""

    # Check image data type
    if not isinstance(image_data, str):
        raise ValueError("Invalid image data.")

    # Check image data URL
    if "," not in image_data:
        raise ValueError("Invalid image format.")

    header, encoded_image = image_data.split(",", 1)

    # Allow only JPEG and PNG
    allowed_headers = [
        "data:image/jpeg;base64",
        "data:image/png;base64"
    ]

    if header.lower() not in allowed_headers:
        raise ValueError(
            "Only JPEG and PNG images are accepted."
        )

    # Check empty image
    if not encoded_image:
        raise ValueError("The captured image is empty.")

    # Check approximate encoded size
    maximum_encoded_size = (
        (MAX_IMAGE_BYTES * 4 // 3) + 8
    )

    if len(encoded_image) > maximum_encoded_size:
        raise ValueError(
            "Image is too large. Maximum size is 5 MB."
        )

    # Strict Base64 decoding
    try:
        image_bytes = base64.b64decode(
            encoded_image,
            validate=True
        )

    except (ValueError, base64.binascii.Error):
        raise ValueError("Invalid Base64 image data.")

    # Check actual decoded size
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            "Image is empty or exceeds the 5 MB limit."
        )

    return image_bytes


def convert_image_to_opencv(image_data):

    """Convert validated camera image data into OpenCV format."""

    image_bytes = decode_camera_image(image_data)

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not process the photograph."
        )

    return image


def extract_visitor_face(image):

    """Detect and extract exactly one visitor face."""

    if not os.path.isfile(CASCADE_FILE):
        raise FileNotFoundError(
            "Face detector file is missing."
        )

    detector = cv2.CascadeClassifier(CASCADE_FILE)

    if detector.empty():
        raise RuntimeError(
            "Face detector could not be loaded."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    detected_faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(detected_faces) == 0:
        raise ValueError(
            "No face detected. Please try again."
        )

    if len(detected_faces) > 1:
        raise ValueError(
            "Multiple faces detected. Only one person should be in the frame."
        )

    x, y, width, height = detected_faces[0]

    face = gray[
        y:y + height,
        x:x + width
    ]

    face = cv2.resize(
        face,
        (200, 200)
    )

    return face


def load_registered_visitor_face():

    """Load and prepare the currently registered visitor's face."""

    # Get temporary photograph filename from session
    filename = session.get("visitor_face_filename")

    if not filename:
        raise ValueError(
            "No registered visitor photograph found."
        )

    # Prevent access to files outside the temporary folder
    if os.path.basename(filename) != filename:
        raise ValueError(
            "Invalid visitor photograph filename."
        )

    # Build photograph path
    image_path = os.path.join(
        TEMP_VISITORS_DIR,
        filename
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            "Registered visitor photograph is missing."
        )

    # Load photograph using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Could not load registered visitor photograph."
        )

    # Extract and prepare the face
    face = extract_visitor_face(image)

    return face


def compare_visitor_faces(registered_face, scanned_face):

    """Compare a scanned face with the registered visitor face."""

    # Create a temporary LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train using the registered visitor's face
    recognizer.train(
        [registered_face],
        np.array([1], dtype=np.int32)
    )

    # Compare the newly scanned face
    predicted_label, confidence = recognizer.predict(
        scanned_face
    )

    distance = round(float(confidence), 2)

    # Check whether the registered visitor matches
    matched = (
        predicted_label == 1
        and distance < MATCH_THRESHOLD
    )

    return matched, distance

def generate_passenger_details(visitor):

    """Generate fictional passenger and flight details."""

    # Generate a unique temporary passenger ID
    passenger_id = "DY-" + ''.join(
        random.choices(string.digits, k=6)
    )

    # Generate fictional flight information
    flight_number = "DY " + str(
        random.randint(100, 999)
    )

    # Generate a departure date and time
    departure_datetime = datetime.now() + timedelta(
        days=random.randint(1, 7)
    )

    flight_date = departure_datetime.strftime("%d %b %Y")
    departure_time = departure_datetime.strftime("%I:%M %p")

    # Generate seat number
    seat_number = (
        str(random.randint(1, 30))
        + random.choice(["A", "B", "C", "D", "E", "F"])
    )

    # Generate boarding gate
    gate_number = "G" + str(random.randint(1, 20))

        # Match common Indian destinations with airport codes
    destination_codes = {
        "mumbai": "BOM",
        "delhi": "DEL",
        "new delhi": "DEL",
        "bengaluru": "BLR",
        "bangalore": "BLR",
        "chennai": "MAA",
        "hyderabad": "HYD",
        "kolkata": "CCU",
        "pune": "PNQ",
        "goa": "GOI",
        "jaipur": "JAI",
        "ahmedabad": "AMD",
        "lucknow": "LKO",
        "kochi": "COK"
    }

    destination_code = destination_codes.get(
        visitor.get("destination", "").strip().lower(),
        "TBD"
    )

    # Prepare passenger details
    passenger_details = {
        "passenger_id": passenger_id,
        "passenger_name": visitor.get("full_name"),
        "destination": visitor.get("destination"),
        "flight_number": flight_number,
        "flight_date": flight_date,
        "departure_time": departure_time,
        "seat_number": seat_number,
        "gate_number": gate_number,
        "origin": "Mumbai",
        "origin_code": "BOM",
        "destination_code": destination_code
    }

    return passenger_details

def generate_boarding_qr(passenger_details):

    """Generate a QR code for the passenger boarding pass."""

    # Create the QR code folder if it does not exist
    os.makedirs(QR_CODES_DIR, exist_ok=True)

    # Prepare QR code data
    qr_data = json.dumps(passenger_details)

    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4
    )

    qr.add_data(qr_data)
    qr.make(fit=True)

    # Create QR code image
    qr_image = qr.make_image(
        fill_color="black",
        back_color="white"
    )

    # Create a unique filename
    filename = passenger_details["passenger_id"] + ".png"

    image_path = os.path.join(
        QR_CODES_DIR,
        filename
    )

    # Save QR code image
    qr_image.save(image_path)

    return filename


# ============================================================
# 5. ADMINISTRATOR LOGIN
# ============================================================

@app.route("/visitor-register", methods=["GET", "POST"])
def visitor_register():

    if request.method == "POST":

        full_name = request.form.get("full_name", "").strip()
        age = request.form.get("age", "").strip()
        destination = request.form.get("destination", "").strip()
        face_consent = request.form.get("face_consent")

        if not full_name or not destination or face_consent != "yes":
            return render_template(
                "visitor_register.html",
                error="Please complete the required fields and give consent."
            )

        session["visitor"] = {
            "full_name": full_name,
            "age": age,
            "destination": destination,
            "face_consent": True
        }

        return redirect(url_for("visitor_face_register"))

    return render_template("visitor_register.html")

@app.route("/visitor-face-register")
def visitor_face_register():

    if "visitor" not in session:
        return redirect(url_for("visitor_register"))

    return render_template("visitor_face_register.html")

@app.route("/visitor-verification-scanner")
def visitor_verification_scanner():

    """Display the visitor face verification scanner."""

    if not session.get("visitor_face_filename"):
        return redirect(url_for("visitor_register"))

    return render_template(
        "visitor_verification_scanner.html"
    )

@app.route("/visitor-passenger-details")
def visitor_passenger_details():

    """Display passenger details after first face verification."""

    visitor = session.get("visitor")

    if not isinstance(visitor, dict):
        return redirect(url_for("visitor_register"))

    if not session.get("visitor_face_filename"):
        return redirect(url_for("visitor_register"))

    if not session.get("first_scan_verified"):
        return redirect(url_for("visitor_verification_scanner"))

    # Generate passenger details only once per visitor session
    passenger_details = session.get("passenger_details")

    if not passenger_details:
        passenger_details = generate_passenger_details(visitor)
        session["passenger_details"] = passenger_details

    return render_template(
        "visitor_passenger_details.html",
        passenger=passenger_details
    )
    
@app.route("/visitor-second-verification")
def visitor_second_verification():

    """Display the second visitor face verification scanner."""

    if not session.get("visitor_face_filename"):
        return redirect(url_for("visitor_register"))

    if not session.get("first_scan_verified"):
        return redirect(url_for("visitor_verification_scanner"))

    return render_template(
        "visitor_second_verification.html"
    )
    
@app.route("/generate-boarding-pass")
def generate_boarding_pass():

    """Generate and display the visitor's simulated boarding pass."""

    visitor = session.get("visitor")

    if not isinstance(visitor, dict):
        return redirect(url_for("visitor_register"))

    if not session.get("first_scan_verified"):
        return redirect(url_for("visitor_verification_scanner"))

    if not session.get("second_scan_verified"):
        return redirect(url_for("visitor_second_verification"))

    # Reuse existing passenger details for this session
    passenger_details = session.get("passenger_details")

    # Generate passenger details if they do not exist
    if not passenger_details:
        passenger_details = generate_passenger_details(visitor)

    # Generate QR code if it has not already been created
    if not passenger_details.get("qr_filename"):

        qr_filename = generate_boarding_qr(passenger_details)

        passenger_details["qr_filename"] = qr_filename

    # Save updated passenger details in the session
    session["passenger_details"] = passenger_details
    
    return render_template(
        "boarding_pass.html",
        passenger=passenger_details
    )

@app.route("/api/register-visitor-face", methods=["POST"])
def register_visitor_face():

    # Check visitor registration and consent
    visitor = session.get("visitor")

    if not isinstance(visitor, dict) or visitor.get("face_consent") is not True:
        return jsonify({
            "success": False,
            "message": "Please complete visitor registration and consent first."
        }), 403

    try:
        # Receive captured image
        data = request.get_json(silent=True)

        if not isinstance(data, dict) or "image" not in data:
            return jsonify({
                "success": False,
                "message": "No valid image received."
            }), 400

        # Convert and validate image
        image = convert_image_to_opencv(data["image"])

        # Create temporary folder if it does not exist
        os.makedirs(TEMP_VISITORS_DIR, exist_ok=True)

        # Generate a random temporary visitor ID
        visitor_id = os.urandom(16).hex()

        # Create a unique image filename
        filename = f"visitor_{visitor_id}.jpg"

        image_path = os.path.join(
            TEMP_VISITORS_DIR,
            filename
        )

        # Save captured image
        saved = cv2.imwrite(image_path, image)

        if not saved:
            raise OSError("Could not save the captured photograph.")

        # Store temporary registration details in the session
        session["visitor_id"] = visitor_id
        session["visitor_face_filename"] = filename

        return jsonify({
            "success": True,
            "visitor_id": visitor_id,
            "message": "Visitor face photograph registered successfully."
        }), 200

    except ValueError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception:
        logger.exception("Visitor face registration error.")

        return jsonify({
            "success": False,
            "message": "Unable to register the photograph. Please try again."
        }), 500

@app.route("/api/verify-visitor-face", methods=["POST"])
def verify_visitor_face():

    """Verify a visitor during the first or second face scan."""

    visitor = session.get("visitor")

    if (
        not isinstance(visitor, dict)
        or visitor.get("face_consent") is not True
        or not session.get("visitor_face_filename")
    ):
        return jsonify({
            "success": False,
            "message": "Please complete visitor registration first."
        }), 403

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError("Invalid scan data.")

        stage = data.get("stage")

        if stage not in ["first", "second"]:
            raise ValueError("Invalid verification stage.")

        if stage == "second" and not session.get("first_scan_verified"):
            return jsonify({
                "success": False,
                "message": "Please complete the first face scan."
            }), 403

        if stage == "first":
            session.pop("first_scan_verified", None)
            session.pop("second_scan_verified", None)

        if "image" not in data:
            raise ValueError("No scanned image received.")

        scanned_image = convert_image_to_opencv(data["image"])

        scanned_face = extract_visitor_face(scanned_image)

        registered_face = load_registered_visitor_face()

        matched, distance = compare_visitor_faces(
            registered_face,
            scanned_face
        )

        if not matched:
            return jsonify({
                "success": False,
                "matched": False,
                "message": "Face not matched. Please try again.",
                "distance": distance
            }), 401

        if stage == "first":
            session["first_scan_verified"] = True

        if stage == "second":
            session["second_scan_verified"] = True

        return jsonify({
            "success": True,
            "matched": True,
            "stage": stage,
            "distance": distance,
            "message": "Face verification successful."
        }), 200

    except ValueError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception:
        logger.exception("Visitor face verification error.")

        return jsonify({
            "success": False,
            "message": "Face verification failed. Please try again."
        }), 500


@app.route("/finish-visit")
def finish_visit():

    """Delete temporary visitor files and clear the session."""

    # Get visitor photograph filename
    visitor_face_filename = session.get("visitor_face_filename")

    # Get boarding pass details
    passenger_details = session.get("passenger_details", {})

    qr_filename = passenger_details.get("qr_filename")

    # Delete visitor photograph
    if visitor_face_filename:

        face_path = os.path.join(
            TEMP_VISITORS_DIR,
            os.path.basename(visitor_face_filename)
        )

        if os.path.isfile(face_path):
            os.remove(face_path)

    # Delete generated QR code
    if qr_filename:

        qr_path = os.path.join(
            QR_CODES_DIR,
            os.path.basename(qr_filename)
        )

        if os.path.isfile(qr_path):
            os.remove(qr_path)
        
    # Clear visitor session
    session.clear()

    # Return to project homepage
    return redirect(url_for("project_home"))


@app.route("/")
def project_home():
    return render_template("project_home.html")




# ============================================================
# 14. APPLICATION START
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
