// UploadPage.jsx
import { useState, useRef, useEffect } from "react";
import { Link } from "react-router";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUpload, faSpinner, faCheck, faTimes, faEnvelope } from "@fortawesome/free-solid-svg-icons";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [progress, setProgress] = useState(0);
    const [apiStatus, setApiStatus] = useState(null);
    const [intervalId, setIntervalId] = useState(null);
    const [email, setEmail] = useState("");
    const [sendEmail, setSendEmail] = useState(false);
    const [cookies] = useCookies(["darkMode"]);
    const dropAreaRef = useRef(null);
    const fileInputRef = useRef(null);

    useEffect(() => {
        return () => intervalId && clearInterval(intervalId);
    }, [intervalId]);

    useEffect(() => {
        const checkApiStatus = async () => {
            try {
                const res = await fetch("/api/health");
                setApiStatus(res.ok ? "connected" : "error");
            } catch {
                setApiStatus("error");
            }
        };
        checkApiStatus();
    }, []);

    const handleImageChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        if (file.type.startsWith("image/")) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
        } else {
            alert("Only image files are allowed.");
        }
    };

    const handleUpload = async () => {
        if (!image || loading) return;
        setLoading(true);
        setTimer(0);
        setProgress(0);
        intervalId && clearInterval(intervalId);

        const id = setInterval(() => {
            setTimer((t) => t + 1);
            setProgress((p) => Math.min(p + 10, 90));
        }, 1000);
        setIntervalId(id);

        const formData = new FormData();
        formData.append("image", image);
        if (sendEmail && email) {
            formData.append("email", email);
        }

        try {
            const res = await fetch(sendEmail ? "/api/predict_and_email" : "/api/predict", {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            setProgress(100);

            setResult({
                status: data.success ? "success" : "error",
                prediction: data.prediction || data.error,
                confidence: data.confidence,
                timestamp: new Date().toLocaleTimeString(),
            });
        } catch (err) {
            setResult({
                status: "error",
                prediction: "Upload failed. Check backend connection.",
            });
        } finally {
            clearInterval(id);
            setLoading(false);
        }
    };

    const isDarkMode = cookies.darkMode === "true";

    return (
        <div className={`flex flex-col items-center justify-center min-h-screen p-4 ${isDarkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
            <motion.div className={`p-6 shadow-lg rounded-2xl w-full max-w-2xl border-8 border-double ${isDarkMode ? "border-blue-600 bg-gray-800" : "border-blue-400 bg-white"}`}
                initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
            >
                <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-sky-600"}`}>
                    Plant Disease Detection
                </h1>

                {apiStatus && (
                    <div className={`text-sm mb-4 p-2 rounded ${apiStatus === "connected" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                        {apiStatus === "connected" ? "Backend connected" : "Backend unavailable"}
                    </div>
                )}

                {/* Upload area */}
                <div ref={dropAreaRef}
                    onDragOver={(e) => e.preventDefault()}
                    onDragLeave={() => dropAreaRef.current?.classList.remove("border-green-500")}
                    onDrop={(e) => {
                        e.preventDefault();
                        const file = e.dataTransfer.files[0];
                        if (file && file.type.startsWith("image/")) handleImageChange({ target: { files: [file] } });
                    }}
                    onClick={() => fileInputRef.current?.click()}
                    className={`w-full p-8 border-2 border-dashed rounded-lg mb-6 text-center cursor-pointer transition-colors ${isDarkMode ? "border-gray-600 hover:border-blue-400" : "border-gray-300 hover:border-blue-500"}`}
                >
                    <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleImageChange} />
                    <FontAwesomeIcon icon={faUpload} className="mx-auto text-4xl mb-3 text-blue-500" />
                    <p className="text-lg">{image ? image.name : "Drag & drop an image or click to select"}</p>
                    <p className="text-sm text-gray-500 mt-2">Supported formats: JPEG, PNG (max 10MB)</p>
                </div>

                {preview && (
                    <div className="w-full mb-4">
                        <img src={preview} alt="Preview" className="w-full max-h-64 object-contain rounded-lg shadow-md mx-auto" />
                        {loading && (
                            <div className="h-2 bg-blue-400 rounded-full mt-2" style={{ width: `${progress}%` }}></div>
                        )}
                    </div>
                )}

                {/* Email Input */}
                <div className="w-full mb-4 flex flex-col gap-2">
                    <label className="flex items-center gap-2">
                        <input type="checkbox" checked={sendEmail} onChange={(e) => setSendEmail(e.target.checked)} />
                        <span>Send result to my email</span>
                    </label>
                    {sendEmail && (
                        <div className="flex items-center border px-3 py-1 rounded-lg bg-white text-black">
                            <FontAwesomeIcon icon={faEnvelope} className="text-gray-400 mr-2" />
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="Enter your email"
                                className="w-full outline-none"
                            />
                        </div>
                    )}
                </div>

                {/* Buttons */}
                <div className="flex gap-4 w-full justify-center mb-4">
                    <button onClick={() => fileInputRef.current?.click()} className="px-4 py-2 rounded-lg bg-gray-300 hover:bg-gray-400">
                        Change Image
                    </button>
                    <button
                        onClick={handleUpload}
                        disabled={loading || !image || apiStatus !== "connected" || (sendEmail && !email)}
                        className="px-6 py-2 rounded-lg flex items-center gap-2 bg-blue-700 hover:bg-blue-600 text-white disabled:opacity-50"
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

                {/* Result */}
                {result && (
                    <motion.div
                        className={`w-full mt-6 p-4 rounded-lg ${
                            result.status === "success" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                        }`}
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    >
                        <p><strong>{result.status === "success" ? "Prediction:" : "Error:"}</strong> {result.prediction}</p>
                        {result.confidence && <p>Confidence: {result.confidence}%</p>}
                        {result.timestamp && <p className="text-xs mt-1 opacity-70">Analyzed at {result.timestamp}</p>}
                    </motion.div>
                )}

                <div className={`w-full mt-8 p-4 rounded-lg ${isDarkMode ? "bg-gray-700" : "bg-gray-200"}`}>
                    <p className="text-center">
                        🌿 Upload a clear photo of your plant's leaves for disease analysis. 
                        For best results, use well-lit photos showing affected areas clearly.
                    </p>
                    <p className="text-center mt-2">
                        Need help interpreting results? Visit our{" "}
                        <Link to="/diagnosis" className="font-bold text-blue-600 hover:text-blue-800">Diagnosis Guide</Link>
                    </p>
                </div>
            </motion.div>
        </div>
    );
}
