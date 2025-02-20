import { useState } from "react";
import { Link } from "react-router";
import Button from "../../components/Button";
import "./Page.css";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState("");
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [intervalId, setIntervalId] = useState(null);

    // Handle Image Selection
    const handleImageChange = (event) => {
        const file = event.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setResult(""); // Reset result when a new image is uploaded
    };

    // Handle Image Upload & AI Analysis
    const handleUpload = async () => {
        if (!image) {
            alert("Please select an image.");
            return;
        }

        setLoading(true);
        setTimer(0); // Reset timer before starting
        setResult("");

        // Start timer
        const newIntervalId = setInterval(() => {
            setTimer((prevTimer) => prevTimer + 1);
        }, 1000);
        setIntervalId(newIntervalId);

        const formData = new FormData();
        formData.append("image", image);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Failed to analyze image.");

            const data = await response.json();
            setResult(data.message); // Display AI result

        } catch (error) {
            setResult("Error: " + error.message);
        } finally {
            clearInterval(newIntervalId); // Stop the timer
            setLoading(false);
        }
    };

    return (
        <div className="p-4">
            <h1 className="text-xl font-bold mb-2">Upload & Analyze Plant Image</h1>

            {/* File Upload */}
            <input type="file" accept="image/*" onChange={handleImageChange} className="upload-btn" />

            {/* Image Preview */}
            {preview && <img src={preview} alt="Preview" className="w-40 mt-2 rounded-md shadow-md" />}

            {/* Upload & Analyze Button */}
            <Button
                onClick={handleUpload}
                className={`bg-blue-500 text-white px-4 py-2 mt-2 ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
                disabled={loading}
            >
                {loading ? "Analyzing..." : "Upload & Analyze"}
            </Button>

            {/* Timer Display */}
            {loading && <p className="text-orange-500 mt-2">Time Elapsed: {timer} sec</p>}

            {/* AI Result */}
            {result && <p className="mt-2 text-green-600 font-semibold">Result: {result}</p>}
        </div>
    );
}
