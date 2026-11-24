
let latestCapturedImage = null;
let cameraStream = null;
let isRecognizing = false;

// Update camera status safely
function updateCameraStatus(message, background, color) {
    const status = document.getElementById("cameraStatus");

    if (!status) {
        console.error("Camera status element not found.");
        return;
    }

    status.innerText = message;

    if (background) status.style.background = background;
    if (color) status.style.color = color;
}

// Start laptop webcam
async function startCamera() {
    const video = document.getElementById("camera");

    if (!video) {
        alert("Camera video element was not found.");
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Camera access is not supported by this browser.");
        return;
    }

    if (cameraStream) {
        console.log("Camera is already running.");
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        console.log("Camera started successfully.");

    } catch (error) {
        console.error("Camera error:", error);
        cameraStream = null;

        alert("Unable to access the camera. Please allow camera permission.");
    }
}

// Stop laptop webcam
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    const video = document.getElementById("camera");

    if (video) {
        video.srcObject = null;
    }

    console.log("Camera stopped.");
}

// Capture photograph from webcam
function captureFace() {
    const video = document.getElementById("camera");
    const canvas = document.getElementById("captureCanvas");

    if (!video || !canvas) {
        alert("Camera or capture canvas element was not found.");
        return;
    }

    if (
        !cameraStream ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        video.videoWidth === 0
    ) {
        alert("Please start the camera first.");
        return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
        alert("Unable to prepare the photograph.");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    const capturedImage = canvas.toDataURL("image/jpeg", 0.90);

    latestCapturedImage = capturedImage;

    // Display captured photograph
    const preview = document.getElementById("capturedPreview");

    if (preview) {
        preview.src = capturedImage;
        preview.style.display = "block";
    }

    // Update status
    updateCameraStatus(
        "FACE CAPTURED SUCCESSFULLY!",
        "#dcfce7",
        "#166534"
    );

    console.log("Face photograph captured successfully.");

    return capturedImage;
}

// Send captured photograph to Flask for recognition
async function recognizeFace() {
    if (isRecognizing) {
        console.log("Recognition request is already running.");
        return;
    }

    if (!latestCapturedImage) {
        alert("Please capture your face first.");
        return;
    }

    isRecognizing = true;

    updateCameraStatus(
        "PROCESSING FACE...",
        "#dbeafe",
        "#1e40af"
    );

    try {
        const response = await fetch("/recognize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: latestCapturedImage
            })
        });

        let result;

        try {
            result = await response.json();
        } catch (error) {
            throw new Error("The server returned an invalid response.");
        }

        if (!response.ok) {
            throw new Error(
                result.message || "Server error: " + response.status
            );
        }

        if (result.success) {
            updateCameraStatus(
                "POTENTIAL IDENTITY MATCH: " +
                result.candidate_id +
                " | LBPH Distance: " +
                result.distance,
                "#dcfce7",
                "#166534"
            );

            // Open passenger details after successful recognition
            setTimeout(() => {
                window.location.href = "/passenger-details";
            }, 1500);

        } else {
            updateCameraStatus(
                (result.message || "No matching face found.") +
                " | Candidate: " +
                (result.candidate_id || "Unknown") +
                " | LBPH Distance: " +
                (result.distance ?? "N/A"),
                "#fee2e2",
                "#b91c1c"
            );
        }

        console.log("Recognition response:", result);

    } catch (error) {
        console.error("Recognition error:", error);

        updateCameraStatus(
            error.message || "Unable to connect to the recognition server.",
            "#fee2e2",
            "#b91c1c"
        );

    } finally {
        isRecognizing = false;
    }
}
