import { useState, useRef } from "react";
import { Link } from "react-router"; // Fixed import
import Button from "../../components/Button";
import { motion } from "framer-motion"; // Fixed import
import { useCookies } from "react-cookie";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null); // Changed to object to store full response
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [intervalId, setIntervalId] = useState(null);
    const [cookies] = useCookies(["darkMode"]);
    const dropAreaRef = useRef(null);

    const handleImageChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null); // Reset result
        }
    };

    const handleDragOver = (event) => {
        event.preventDefault();
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleImageChange({ target: { files: [file] } });
        }
    };

    const handleUpload = async () => {
        if (!image) {
            alert("Please select an image.");
            return;
        }

        setLoading(true);
        setTimer(0);
        const newIntervalId = setInterval(() => setTimer(prev => prev + 1), 1000);
        setIntervalId(newIntervalId);

        try {
            const formData = new FormData();
            formData.append("image", image);

            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Analysis failed");
            
            const data = await response.json();
            setResult(data); // Store full response

        } catch (error) {
            setResult({ 
                status: "error",
                prediction: error.message 
            });
        } finally {
            clearInterval(newIntervalId);
            setLoading(false);
        }
    };

    return (
        <div className={`flex flex-col items-center justify-center align-middle w-full h-screen ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
            <motion.div
                className={`p-6 shadow-lg rounded-2xl w-3/5 h-[72vh] flex flex-col justify-center items-center border-8 border-double border-blue-700`}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <h1 className={`text-2xl font-bold mb-4 ${cookies.darkMode ? "text-white" : "text-sky-600"}`}>
                    Upload & Analyze Plant Image
                </h1>

                <div className="flex flex-col items-center">
                    <label htmlFor="file-upload" className={`cursor-pointer rounded-xl px-6 py-2 border mb-4 ${cookies.darkMode ? "bg-blue-700 hover:bg-blue-500 border-blue-900" : "bg-gray-100 hover:bg-gray-200 border-gray-400"}`}>
                        Choose File
                        <input 
                            type="file" 
                            id="file-upload" 
                            accept="image/*" 
                            onChange={handleImageChange} 
                            className="hidden" 
                        />
                    </label>
                    {image && <p className="text-sm italic">{image.name}</p>}
                    {loading && <p className="text-green-600">Analyzing Image ... {timer} seconds</p>}
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
                    disabled={loading}
                    className={`bg-blue-700 rounded-lg px-5 py-2 transition-all duration-300 hover:bg-blue-500 text-white mt-4 ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                    {loading ? "Analyzing..." : "Upload"}
                </button>

                {result && (
                    <div className={`mt-4 p-3 rounded-lg ${result.status === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                        <p className="font-semibold">
                            {result.status === "success" ? "✅ Analysis Result" : "❌ Error"}
                        </p>
                        <p>{result.prediction}</p>
                        {result.confidence && (
                            <p className="text-sm mt-1">Confidence: {result.confidence}</p>
                        )}
                    </div>
                )}

                <div className="w-full text-center mt-8">
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