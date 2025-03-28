import { useState, useRef } from "react";
import { Link } from "react-router";
import Button from "../../components/Button";
import * as motion from "motion/react-client";
import React from "react";
import { useCookies } from "react-cookie";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState("");
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [intervalId, setIntervalId] = useState(null);
    const [cookies] = useCookies(["darkMode"]);
    const dropAreaRef = useRef(null);

    // Handle Image Selection
    const handleImageChange = (event) => {
        const file = event.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setResult(""); // Reset result when a new image is uploaded
    };
    // Handle image selection
    const handleFileInputChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            handleImageChange(file);
        }
    };

    //Handle drag over
    const handleDragOver = (event) => {
        event.preventDefault();
    }

    //Handle drop
    const handleDrop = (event) => {
        event.preventDefalt();
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleImageChange(file);
        }
    }

    // Handle Image Upload & AI Analysis
    const handleUpload = async () => {
        if (!image) {
            alert("Please select an image.");
            return;
        }

        alert("Image sent to admin: ", image);

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
        <div className={`flex flex-col items-center justify-center align-middle w-full h-screen ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
            <motion.div
                className={`p-6 shadow-lg rounded-2xl w-3/5 h-[72vh] flex flex-col justify-center items-center border border-blue-700`}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <motion.h1
                    className={`text-2xl font-bold mb-4 ${cookies.darkMode ? "text-white" : "text-blue-900"}`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    <h1>Upload & Analyze Plant Image</h1>
                </motion.h1>

                <div>
                    <input type="file"
                        accept="image/*"
                        onChange={handleImageChange}
                        className="w-300 border border-blue-700 p-3 rounded-lg shadow-sm mb-4 cursor-pointer"
                    />
                    {loading && <p>Analyzing Image ... {timer} seconds</p>}
                </div>

                

                {preview && (
                    <motion.img
                        src={preview}
                        alt="Preview"
                        className="w-80 mt-8 rounded-lg shadow-md mx-auto"
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.3 }}
                        whileHover={{ scale: 1.05 }}
                    />
                )}

                <button
                    onClick={handleUpload}
                    className={`bg-blue-700 rounded-lg px-5 py-2 transition-all duration-300 hover:bg-blue-500 text-white mt-4 ${loading ? "opacity-50 cursor-not-allowed" : ""
                        }`}
                >
                    Upload
                </button>

                {result && (
                    <p
                        className={`mt-2 font-semibold ${result.includes("Healthy")
                            ? "text-green-600"
                            : "text-orange-400"
                            }`}
                    >
                        Result: {result}
                    </p>
                )}

                <div className="w-full text-center mt-12">
                    <p className="text-lg">
                        🌿 Need help identifying plant diseases? Upload an image and
                        let AI analyze it instantly! If you need expert advice, visit
                        our{" "}
                        <Link
                            to="/diagnosis"
                            className="text-blue-500 hover:text-blue-700 underline"
                        >
                            Diagnosis Page
                        </Link>
                        .
                    </p>
                </div>
            </motion.div>
        </div>
    );
}