import { useState, useRef } from "react";
import { Link } from "react-router-dom"; // Correct import for React Router v6
import { motion } from "framer-motion"; // Use framer-motion instead of motion/react-client
import { useCookies } from "react-cookie";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState("");
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [intervalId, setIntervalId] = useState(null);
    const [cookies] = useCookies(["darkMode"]);
    const dropAreaRef = useRef(null); // Create a reference for the drop area

    // Handle Image Selection (File Input or Drag & Drop)
    const handleImageChange = (file) => {
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setResult("");
    };

    // Handle File Input Change
    const handleFileInputChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            handleImageChange(file);
        }
    };

    // Handle Drag Over
    const handleDragOver = (event) => {
        event.preventDefault();
    };

    // Handle Drop
    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleImageChange(file);
        }
    };

    // Handle Image Upload & AI Analysis
    const handleUpload = async () => {
        if (!image) {
            alert("Please select an image.");
            return;
        }

        setLoading(true);
        setTimer(0);
        setResult("");

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

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || "Failed to analyze image.");
            }

            const data = await response.json();
            setResult(data.message);
        } catch (error) {
            setResult("Error: " + error.message);
        } finally {
            clearInterval(newIntervalId);
            setLoading(false);
        }
    };

    return (
        <div className={`flex flex-col items-center justify-center align-middle w-full h-full ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
            <motion.div
                className="p-6 shadow-lg rounded-2xl w-3/5 h-[72vh] flex flex-col justify-center items-center border border-blue-700"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <motion.h1
                    className="text-2xl font-bold mb-4 text-blue-800"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                >
                    Upload & Analyze Plant Image
                </motion.h1>

                <div
                    ref={dropAreaRef}
                    className="border-2 border-dashed border-gray-400 p-8 rounded-md w-80 h-48 flex flex-col items-center justify-center cursor-pointer mb-4"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                >
                    {preview ? (
                        <img src={preview} alt="Preview" className="w-full h-full object-contain" />
                    ) : (
                        <p className="text-gray-600">Drag and drop image here or click to select</p>
                    )}
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileInputChange}
                        className="hidden"
                    />
                </div>

                {loading && <p>Analyzing Image ... {timer} seconds</p>}
