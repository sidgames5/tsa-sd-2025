import { useState } from "react";
import { Link } from "react-router";
import Button from "../../components/Button";
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
            className="p-6 bg-gray-900 shadow-lg rounded-2xl w-3/5 h-2/5 mx-auto flex flex-col justify-center items-center mt-[15vh] border border-green-300"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <motion.h1
                className="text-2xl font-bold mb-4 text-green-800"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
            >
                <h1>Upload & Analyze Plant Image</h1>
            </motion.h1>

            <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="w-full border border-green-400 p-3 rounded-lg bg-gray-950 shadow-sm mb-4 cursor-pointer text-white"
            />

            {/* Image Preiew Placeholder */}
            {!preview && (
                <div className="w-80 h-52 border-2 border-dashed border-gray-600 rounded-lg flex justify-center items-center mb-4 text-white">Image Here</div>
            )}

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

            <motion.div
                className="flex justify-center mt-4"
                whileHover={{ scale: 1.05 }}
            >
                <Button
                    onClick={handleUpload}
                    className={`bg-blue-500 text-white px-5 py-2 mt- transition hover:bg-blue-700 ${loading ? "opacity-50 cursor-not-allowed" : ""
                        }`}
                >
                    Upload
                </Button>
            </motion.div>

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
                <p className="text-lg text-white">
                    🌿 Need help identifying plant diseases? Upload an image and
                    let AI analyze it instantly! If you need expert advice, visit
                    our{" "}
                    <Link
                        to="/diagnosis"
                        className="text-blue-500 hover:underline"
                    >
                        Diagnosis Page
                    </Link>
                    .
                </p>
            </div>
        </motion.div>
    );
}