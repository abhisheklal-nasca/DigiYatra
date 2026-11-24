
let secondCapturedImage = null;
let isVerifyingSecondFace = false;

// Update verification status safely
function updateVerificationStatus(message, background, color) {
    const status = document.getElementById("cameraStatus");

    if (!status) {
        console.error("Verification status element not found.");
        return;
    }

    status.innerText = message;

    if (background) status.style.background = background;
    if (color) status.style.color = color;
}

// Capture a fresh image for airport verification
function captureSecondFace() {
    const video = document.getElementById("camera");
    const canvas = document.getElementById("captureCanvas");

    if (!video || !canvas) {
        alert("Camera or capture canvas element was not found.");
        return;
    }

    if (
        typeof cameraStream === "undefined" ||
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

    secondCapturedImage = canvas.toDataURL("image/jpeg", 0.90);

    // Display captured photograph
    const preview = document.getElementById("capturedPreview");

    if (preview) {
        preview.src = secondCapturedImage;
        preview.style.display = "block";
    }

    updateVerificationStatus(
        "SECOND FACE CAPTURED SUCCESSFULLY!",
        "#dcfce7",
        "#166534"
    );

    console.log("Second verification photograph captured.");

    return secondCapturedImage;
}

// Send second photograph to Python
async function verifySecondFace() {
    if (isVerifyingSecondFace) {
        console.log("Verification request is already running.");
        return;
    }

    if (!secondCapturedImage) {
        alert("Please capture your face first.");
        return;
    }

    isVerifyingSecondFace = true;

    updateVerificationStatus(
        "VERIFYING YOUR IDENTITY...",
        "#e0f2fe",
        "#075985"
    );

    try {
        const response = await fetch("/verify-second-face", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: secondCapturedImage
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
            updateVerificationStatus(
                result.message || "Second verification successful.",
                "#dcfce7",
                "#166534"
            );

            console.log("Second verification successful:", result);

            setTimeout(() => {
                window.location.href = "/verification-result";
            }, 1500);

        } else {
            updateVerificationStatus(
                (result.message || "Second verification failed.") +
                " | Candidate: " +
                (result.candidate_id || "Unknown") +
                " | Distance: " +
                (result.distance ?? "N/A"),
                "#fee2e2",
                "#b91c1c"
            );

            console.log("Second verification rejected:", result);
        }

    } catch (error) {
        console.error("Verification error:", error);

        updateVerificationStatus(
            error.message || "Unable to connect to the verification server.",
            "#fee2e2",
            "#b91c1c"
        );

    } finally {
        isVerifyingSecondFace = false;
    }
}
