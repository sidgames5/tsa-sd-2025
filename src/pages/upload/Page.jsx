import { useState } from "react";
import { Link } from "react-router";
import Button from "../../components/Button";
import "./Page.css";
import * as motion from "motion/react-client";
import React from "react";

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
        <motion.div
            className="p-6 bg-white shadow-lg rounded-xl w-full max-w-md mx-auto flex flex-col justify-center items-center"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}>
            <motion.h1
                className="text-2xl font-bold mb-4 text-gray-800"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}>

                <h1>Upload & Analyze Plant Image</h1>
            </motion.h1>

            <input type="file" accept="image/*" onChange={handleImageChange} className="w-full border p-2 rounded-md mb-4 cursor-pointer" />

            {preview && (
                <motion.img src={preview} alt="Preview" className="w-40 mt-2 rounded-lg shadow-md mx-auto" initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.3 }} whileHover={{ scale: 1.05 }} />
            )}

            <motion.div className="flex justify-center mt-4" whileHover={{ scale: 1.05 }}>
                <Button onClick={handleUpload} className={`bg-blue-500 text-white px-4 py-2 mt-2 ${loading ? "opacity-50 cursor-not-allowed" : ""}`}>Upload</Button>
            </motion.div>

            {result && <p className={`mt-2 font-semibold ${(function () {
                switch (result) {
                    case "Detected: Healthy":
                        return "text-green-600";
                    default:
                        return "text-orange-400";
                }
            })()}`}>Result: {result}</p>}
        </motion.div>
        // <div className="p-4">
        //     <h1 className="text-xl font-bold mb-2">Upload & Analyze Plant Image</h1>

        //     {/* File Upload */}
        //     /*<input type="file" accept="image/*" onChange={handleImageChange} className="upload-btn" />

        //     {/* Image Preview */}
        //     /*preview && <img src={preview} alt="Preview" className="w-40 mt-2 rounded-md shadow-md" />}

        //     {/* Upload & Analyze Button */}
        //     /*<Button
        //         onClick={handleUpload}
        //         className={`bg-blue-500 text-white px-4 py-2 mt-2 ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
        //         disabled={loading}
        //     >
        //         {loading ? "Analyzing..." : "Upload & Analyze"}
        //     </Button>

        //     {/* Timer Display */}
        //     /*loading && <p className="text-orange-500 mt-2">Time Elapsed: {timer} sec</p>}

        //     {/* AI Result */}
        //     /*{result && <p className="mt-2 text-green-600 font-semibold">Result: {result}</p>}
        // </div>
    );
}
