import { useState, useRef, useEffect } from "react";
import { Link } from "react-router";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUpload, faSpinner, faCheck, faTimes } from "@fortawesome/free-solid-svg-icons";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [progress, setProgress] = useState(0);
    const [apiStatus, setApiStatus] = useState(null);
    const [intervalId, setIntervalId] = useState(null);
    const [cookies] = useCookies(["darkMode"]);
    const dropAreaRef = useRef(null);
    const fileInputRef = useRef(null);

    // Clean up interval on component unmount
    useEffect(() => {
        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [intervalId]);

    // Check backend connection
    useEffect(() => {
        const checkApiStatus = async () => {
            try {
                const response = await fetch("/api/health");
                setApiStatus(response.ok ? "connected" : "error");
            } catch (error) {
                setApiStatus("error");
                console.error("Backend connection error:", error);
            }
        };
        checkApiStatus();
    }, []);

    const handleImageChange = (event) => {
        try {
            const file = event.target.files?.[0];
            if (!file) return;
            
            if (file.type.startsWith("image/")) {
                setImage(file);
                setPreview(URL.createObjectURL(file));
                setResult(null);
            } else {
                alert("Please upload an image file (JPEG, PNG)");
            }
        } catch (error) {
            console.error("Image handling error:", error);
        }
    };

    const handleDragOver = (event) => {
        event.preventDefault();
        if (dropAreaRef.current) {
            dropAreaRef.current.classList.add("border-green-500");
        }
    };

    const handleDragLeave = () => {
        if (dropAreaRef.current) {
            dropAreaRef.current.classList.remove("border-green-500");
        }
    };

    const handleDrop = (event) => {
        event.preventDefault();
        handleDragLeave();
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleImageChange({ target: { files: [file] } });
        }
    };

    const handleUpload = async () => {
        if (!image || loading) return;

        try {
            // Setup loading state
            setLoading(true);
            setTimer(0);
            setProgress(0);
            
            // Clear any existing interval
            if (intervalId) clearInterval(intervalId);
            
            // Start progress timer
            const newIntervalId = setInterval(() => {
                setTimer(prev => prev + 1);
                setProgress(prev => Math.min(prev + 10, 90));
            }, 1000);
            setIntervalId(newIntervalId);

            // Prepare form data
            const formData = new FormData();
            formData.append("image", image);

            // API call
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            setProgress(100);
            setResult({
                status: data.success ? "success" : "error",
                prediction: data.prediction || data.error,
                confidence: data.confidence,
                timestamp: new Date().toLocaleTimeString()
            });

        } catch (error) {
            console.error("Upload error:", error);
            setResult({ 
                status: "error",
                prediction: error.message.includes("Failed to fetch") 
                    ? "Backend connection failed. Please try again later." 
                    : error.message
            });
        } finally {
            if (intervalId) clearInterval(intervalId);
            setLoading(false);
        }
    };

    const triggerFileInput = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    // Safely get dark mode status
    const isDarkMode = cookies.darkMode === "true";

    return (
        <div className={`flex flex-col items-center justify-center min-h-screen p-4 ${isDarkMode ? "bg-gray-900 text-white" : "bg-neutral-50 text-black"}`}>
            <motion.div
                className={`p-6 max-h-[80vh] mt-10 shadow-lg rounded-2xl w-full max-w-2xl flex flex-col justify-center items-center border-8 border-double ${isDarkMode ? "border-blue-600 bg-gray-800" : "border-blue-400 bg-white"}`}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-sky-600"}`}>
                    Plant Disease Detection
                </h1>

                {/* API Status Indicator */}
                {apiStatus && (
                    <div className={`text-sm mb-4 p-2 rounded ${
                        apiStatus === "connected" ? "bg-green-100 text-green-800" : 
                        apiStatus === "error" ? "bg-red-100 text-red-800" : 
                        "bg-yellow-100 text-yellow-800"
                    }`}>
                        {apiStatus === "connected" ? (
                            "Backend connected"
                        ) : apiStatus === "error" ? (
                            "Backend unavailable - Analysis won't work"
                        ) : (
                            "Checking backend connection..."
                        )}
                    </div>
                )}
                {/* Drag and Drop Area */}

                {!image && (
                    <div
                        ref={dropAreaRef}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={triggerFileInput}
                        className={`w-full p-8 border-2 border-dashed rounded-lg mb-6 text-center cursor-pointer transition-colors ${
                            isDarkMode ? "border-gray-600 hover:border-blue-400" : "border-gray-300 hover:border-blue-500"
                        }`}
                    >
                        <input 
                            type="file" 
                            ref={fileInputRef}
                            accept="image/*" 
                            onChange={handleImageChange} 
                            className="hidden" 
                        />
                        <FontAwesomeIcon 
                            icon={faUpload} 
                            className="mx-auto text-4xl mb-3 text-blue-500" 
                        />
                        <p className="text-lg">
                            {image ? image.name : "Drag & drop an image or click to select"}
                        </p>
                        <p className="text-sm text-gray-500 mt-2">
                            Supported formats: JPEG, PNG (max 10MB)
                        </p>
                    </div>
                )}


                {/* Preview and Progress */}
                {preview && (
                    <motion.div
                        className="w-full mb-6"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="relative">
                            <img
                                src={preview}
                                alt="Preview"
                                className="w-full max-h-64 object-contain rounded-lg shadow-md mx-auto max-h-40"
                                onError={() => setPreview(null)}
                            />
                            {loading && (
                                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2">
                                    <div 
                                        className="h-2 bg-blue-400 rounded-full" 
                                        style={{ width: `${progress}%` }}
                                    ></div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}

                <div className="flex gap-4 w-full justify-center">
                    <button
                        onClick={triggerFileInput}
                        className={`px-4 py-2 rounded-lg ${
                            isDarkMode ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-200 hover:bg-gray-300"
                        }`}
                    >
                        Change Image
                    </button>
                    <button
                        onClick={handleUpload}
                        disabled={loading || !image || apiStatus !== "connected"}
                        className={`px-6 py-2 rounded-lg flex items-center gap-2 ${
                            loading ? "bg-blue-600" : "bg-blue-700 hover:bg-blue-600"
                        } text-white ${(!image || apiStatus !== "connected") ? "opacity-50 cursor-not-allowed" : ""}`}
                    >
                        {loading ? (
                            <>
                                <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                                Analyzing ({timer}s)
                            </>
                        ) : (
                            <>
                                <FontAwesomeIcon icon={faUpload} />
                                Analyze Now
                            </>
                        )}
                    </button>
                </div>

                {/* Results Display */}
                {result && (
                    <motion.div
                        className={`w-full mt-6 p-4 rounded-lg ${
                            result.status === "success" ? 
                                (isDarkMode ? "bg-green-900 text-green-100" : "bg-green-100 text-green-800") : 
                                (isDarkMode ? "bg-red-900 text-red-100" : "bg-red-100 text-red-800")
                        }`}
                        initial={{ scale: 0.9 }}
                        animate={{ scale: 1 }}
                    >
                        <div className="flex items-start gap-3">
                            <FontAwesomeIcon 
                                icon={result.status === "success" ? faCheck : faTimes} 
                                className={`text-2xl mt-1 ${
                                    result.status === "success" ? "text-green-500" : "text-red-500"
                                }`} 
                            />
                            <div>
                                <h3 className="font-bold text-lg">
                                    {result.status === "success" ? "Analysis Complete" : "Analysis Failed"}
                                </h3>
                                <p className="mt-1">{result.prediction}</p>
                                {result.confidence && (
                                    <p className="text-sm mt-2">
                                        Confidence: <span className="font-bold">{result.confidence}%</span>
                                    </p>
                                )}
                                {result.timestamp && (
                                    <p className="text-xs mt-2 opacity-70">
                                        Analyzed at {result.timestamp}
                                    </p>
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}

                {/* Information Section */}
                <div className={`w-full mt-8 p-4 rounded-lg ${
                    isDarkMode ? "bg-gray-700" : "bg-gray-200"
                }`}>
                    <p className="text-center">
                        🌿 Upload a clear photo of your plant's leaves for disease analysis. 
                        For best results, use well-lit photos showing affected areas clearly.
                    </p>
                    <p className="text-center mt-2">
                        Need help interpreting results? Visit our{" "}
                        
                    </p>
                    <p className="text-center mt-2">
                        Don't know what plants our AI uses? To find out, visit our{" "}
                        <Link
                            to="/"
                            className={`font-bold ${
                                isDarkMode ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-800"
                            }`}
                        >
                            Home Page
                        </Link>

                    </p>
                </div>
            </motion.div>
        </div>
    );
}
