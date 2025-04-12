import { useState, useRef, useEffect } from "react";
import { Link, useNavigate } from "react-router";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUpload, faSpinner, faCheck, faTimes, faUser } from "@fortawesome/free-solid-svg-icons";
import { updateUserChartData, getUserChartData} from '../results/chartStuff';
export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [timer, setTimer] = useState(0);
    const [progress, setProgress] = useState(0);
    const [apiStatus, setApiStatus] = useState(null);
    const [intervalId, setIntervalId] = useState(null);
    const [cookies, setCookie] = useCookies(["darkMode", "user"]);
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [authMode, setAuthMode] = useState("login");
    const [authEmail, setAuthEmail] = useState("");
    const [authPassword, setAuthPassword] = useState("");
    const [authConfirmPassword, setAuthConfirmPassword] = useState("");
    const [authMessage, setAuthMessage] = useState("");
    const [showAuthPassword, setShowAuthPassword] = useState(false);
    const dropAreaRef = useRef(null);
    const fileInputRef = useRef(null);
    const navigate = useNavigate();

    // Check if user is logged in
    const isLoggedIn = !!cookies.user;

    useEffect(() => {
        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [intervalId]);

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
        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }

        const fileInput = event.target;
        const file = fileInput.files?.[0];
        fileInput.value = null;

        if (!file) return;

        if (file.type.startsWith("image/")) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
        } else {
            alert("Please upload an image file (JPEG, PNG)");
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
        
        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }

        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) {
            handleImageChange({ target: { files: [file] } });
        }
    };

    const handleUpload = async () => {
        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }
    
        if (!image || loading) return;
    
        try {
            setLoading(true);
            setTimer(0);
            setProgress(0);
            if (intervalId) clearInterval(intervalId);
    
            const newIntervalId = setInterval(() => {
                setTimer((prev) => prev + 1);
                setProgress((prev) => Math.min(prev + 10, 90));
            }, 1000);
            setIntervalId(newIntervalId);
    
            const formData = new FormData();
            formData.append("image", image);
    
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });
    
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
    
            const data = await response.json();
            setProgress(100);
            
          
            const now = new Date();
            const newResult = {
                status: data.success ? "success" : "error",
                prediction: data.prediction || data.error,
                confidence: data.confidence,
                timestamp: now.toLocaleString(),
                image: preview,
                date: now.toISOString().split('T')[0],
                id: Date.now()
            };
            
            setResult(newResult);
    
            
            if (isLoggedIn && cookies.user?.email && data.success) {
                const userEmail = cookies.user.email;
                
                
                const userResults = JSON.parse(localStorage.getItem(`plantResults_${userEmail}`)) || [];
                userResults.unshift(newResult);
                localStorage.setItem(`plantResults_${userEmail}`, JSON.stringify(userResults));
                
            
                const accuracy = data.confidence / 100; // Convert to 0-1 range
                const loss = 1 - accuracy;
                updateUserChartData(userEmail, accuracy, loss);
                
                
                const updatedChartData = getUserChartData(userEmail);
                
            }
        } catch (error) {
            console.error("Upload error:", error);
            setResult({
                status: "error",
                prediction: error.message.includes("Failed to fetch")
                    ? "Backend connection failed. Please try again later."
                    : error.message,
            });
        } finally {
            if (intervalId) clearInterval(intervalId);
            setLoading(false);
        }
    };

    const triggerFileInput = () => {
        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }

        if (fileInputRef.current) {
            fileInputRef.current.value = null;
            fileInputRef.current.click();
        }
    };

    const handleAuthSubmit = (e) => {
        e.preventDefault();

        if (!authEmail || !authPassword || (authMode === "signup" && !authConfirmPassword)) {
            setAuthMessage("Please fill all fields.");
            return;
        }

        if (authMode === "signup") {
            if (authPassword !== authConfirmPassword) {
                setAuthMessage("Passwords do not match!");
                return;
            }

            const existingUser = localStorage.getItem(authEmail);
            if (existingUser) {
                setAuthMessage("User already exists. Please log in.");
                return;
            }

            localStorage.setItem(authEmail, JSON.stringify({ password: authPassword }));
            setAuthMessage("Account created! Please log in.");
            setAuthMode("login");
        } else {
            const storedUser = localStorage.getItem(authEmail);
            if (!storedUser) {
                setAuthMessage("User not found. Please sign up.");
                return;
            }

            const { password: storedPassword } = JSON.parse(storedUser);
            if (authPassword === storedPassword) {
                setAuthMessage("Login successful!");
                setCookie("user", { email: authEmail }, { path: "/" });
                setShowAuthModal(false);
            } else {
                setAuthMessage("Incorrect password.");
            }
        }
    };

    const handleLogout = () => {
        setCookie("user", "", { path: "/", expires: new Date(0) });
        setImage(null);
        setPreview(null);
        setResult(null);
    };

    const toggleAuthMode = () => {
        setAuthMode(authMode === "login" ? "signup" : "login");
        setAuthMessage("");
    };

    const isDarkMode = cookies.darkMode === true;

    return (
        <div className={`flex flex-col items-center justify-center min-h-screen p-4 ${
            isDarkMode
                ? "bg-gradient-to-b from-gray-950 via-black to-gray-900 text-white"
                : "bg-gradient-to-b from-slate-50 via-white to-slate-200 text-gray-900"
        }`}>
            <motion.div
                className={`p-6 max-h-[80vh] mt-10 rounded-2xl w-full max-w-2xl flex flex-col justify-center items-center border-[2px] shadow-xl ring-1 ring-offset-2 ${
                    isDarkMode
                        ? "bg-gradient-to-br from-gray-800 via-gray-900 to-black border-indigo-800 shadow-indigo-900/40 ring-indigo-500/20"
                        : "bg-white border-blue-300 shadow-blue-200/40 ring-blue-400/20"
                }`}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <div className="flex justify-between w-full mt-2">
                    <h1 className={`text-3xl font-bold mb-6 ${isDarkMode ? "text-white" : "text-sky-600"}`}>Plant Disease Detection 🌱</h1>
                    {isLoggedIn ? (
                        <button
                            onClick={handleLogout}
                            className={`flex items-center gap-2 px-3 py-1 rounded-lg text-sm ${
                                isDarkMode
                                    ? "bg-gray-700 hover:bg-gray-600 text-white"
                                    : "bg-gray-200 hover:bg-gray-300 text-gray-800"
                            }`}
                        >
                            <FontAwesomeIcon icon={faUser} />
                            Logout
                        </button>
                    ) : (
                        <button
                            onClick={() => setShowAuthModal(true)}
                            className={`flex items-center gap-2 px-3 py-1 rounded-lg text-sm ${
                                isDarkMode
                                    ? "bg-blue-700 hover:bg-blue-600 text-white"
                                    : "bg-blue-600 hover:bg-blue-500 text-white"
                            }`}
                        >
                            <FontAwesomeIcon icon={faUser} />
                            Login
                        </button>
                    )}
                </div>

                {apiStatus && (
                    <div className={`text-sm mb-4 p-2 rounded ${
                        apiStatus === "connected"
                            ? "bg-green-100 text-green-800"
                            : apiStatus === "error"
                            ? "bg-red-100 text-red-800"
                            : "bg-yellow-100 text-yellow-800"
                    }`}>
                        {apiStatus === "connected"
                            ? "Backend connected"
                            : apiStatus === "error"
                            ? "Backend unavailable - Analysis won't work"
                            : "Checking backend connection..."}
                    </div>
                )}

                {!isLoggedIn && (
                    <div className={`w-full p-8 mb-6 text-center rounded-lg ${
                        isDarkMode ? "bg-gray-900/80 text-gray-300" : "bg-gray-100 text-gray-800"
                    }`}>
                        <p className="text-lg font-medium mb-2">Please login to upload and analyze plant images</p>
                        <button
                            onClick={() => setShowAuthModal(true)}
                            className={`px-4 py-2 rounded-lg ${
                                isDarkMode
                                    ? "bg-blue-600 hover:bg-blue-500 text-white"
                                    : "bg-blue-500 hover:bg-blue-600 text-white"
                            }`}
                        >
                            Login or Sign Up
                        </button>
                    </div>
                )}

                {isLoggedIn && !image && (
                    <div
                        ref={dropAreaRef}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={triggerFileInput}
                        className={`w-full p-8 border-2 border-dashed rounded-lg mb-6 text-center cursor-pointer transition-all duration-300 ${
                            isDarkMode
                                ? "border-gray-700 hover:border-indigo-500 hover:shadow-md hover:shadow-indigo-700/30 bg-gray-900 text-gray-300"
                                : "border-gray-300 hover:border-blue-500 hover:shadow-md hover:shadow-blue-300/50 bg-white"
                        }`}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            accept="image/*"
                            onChange={handleImageChange}
                            className="hidden"
                        />
                        <FontAwesomeIcon icon={faUpload} className="mx-auto text-4xl mb-3 text-blue-500" />
                        <p className="text-lg">
                            {image ? image.name : "Drag & drop an image or click to select"}
                        </p>
                        <p className="text-sm text-gray-500 mt-2">Supported formats: JPEG, PNG (max 10MB)</p>
                    </div>
                )}

                {preview && (
                    <motion.div className="w-full mb-6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
                        <div className="relative">
                            <img
                                src={preview}
                                alt="Preview"
                                className="w-full max-h-64 object-contain rounded-lg shadow-md mx-auto h-[160px]"
                                onError={() => setPreview(null)}
                            />
                            {loading && (
                                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2">
                                    <div className="h-2 bg-blue-400 rounded-full" style={{ width: `${progress}%` }}></div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}

                {isLoggedIn && (
                    <div className="flex gap-4 w-full justify-center">
                        <button
                            onClick={triggerFileInput}
                            className={`px-4 py-2 rounded-lg shadow-sm transition-all ring-1 ring-offset-1 ${
                                isDarkMode
                                    ? "bg-gray-800 hover:bg-gray-700 text-white ring-gray-600"
                                    : "bg-gray-200 hover:bg-gray-300 text-black ring-gray-300"
                            }`}
                        >
                            Change Image
                        </button>
                        <button
                            onClick={handleUpload}
                            disabled={loading || !image || apiStatus !== "connected"}
                            className={`px-6 py-2 rounded-lg flex items-center gap-2 shadow-md ring-1 ring-offset-2 ${
                                loading
                                    ? "bg-blue-600 ring-blue-700"
                                    : "bg-blue-700 hover:bg-blue-600 ring-blue-600"
                            } text-white ${
                                !image || apiStatus !== "connected"
                                    ? "opacity-50 cursor-not-allowed"
                                    : ""
                            }`}
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
                )}

                {/* Modal for Results */}
                {result && (
                    <div className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-50">
                        <motion.div
                            className={`w-full max-w-lg p-6 rounded-2xl shadow-xl ring-1 ring-offset-2 ${
                                result.status === "success"
                                    ? isDarkMode
                                        ? "bg-gradient-to-br from-slate-700 via-stone-600 to-emerald-500 text-green-100 shadow-lg ring-green-500/50"
                                        : "bg-gradient-to-br from-teal-600 via-sky-500 to-emerald-500 text-white shadow-lg ring-emerald-200/50"
                                    : isDarkMode
                                    ? "bg-gradient-to-br from-red-700 via-stone-600 to-red-900 text-red-100 shadow-lg ring-red-500/30"
                                    : "bg-gradient-to-br from-red-600 via-amber-500 to-orange-500 text-white shadow-lg ring-red-300/20"
                            }`}
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            <div className="flex justify-between items-center mb-4">
                            <h2 className={`text-xl font-bold ${ isDarkMode ? "text-white" : result.status === "success" ? "bg-gradient-to-r from-sky-300 to-blue-500 bg-clip-text text-transparent drop-shadow-md" : "text-amber-50 drop-shadow-md"}`}>
                                    Prediction Result
                                </h2>
                                <button
                                    onClick={() => setResult(null)}
                                    className={`p-2 rounded-full transition ${
                                        isDarkMode 
                                            ? "text-gray-300 hover:bg-gray-700 hover:text-white" 
                                            : result.status === "success"
                                                ? "text-amber-100 hover:bg-emerald-700/80 hover:text-white"
                                                : "text-amber-100 hover:bg-amber-700/80 hover:text-white"
                                    }`}
                                >
                                    <FontAwesomeIcon className="p-2 pb-1" icon={faTimes} />
                                </button>
                            </div>
                            <div className="text-center">
                                {result.status === "success" ? (
                                    <>
                                        <p className={`text-2xl font-bold mb-3 ${isDarkMode ? "text-amber-300 drop-shadow-md" : "text-amber-300 drop-shadow-md"}`}>{result.prediction}</p>
                                        <p className={`${cookies.darkMode ? "text-emerald-200" : "text-emerald-250"} text-sm`}>Confidence: {result.confidence}%</p>
                                        <p className={`${cookies.darkMode ? "text-gray-400" : "text-gray-200"} text-sm`}>Timestamp: {result.timestamp}</p>
                                    </>
                                ) : (
                                    <p className={`text-lg font-semibold ${
                                        isDarkMode 
                                            ? "text-amber-200" 
                                            : "text-amber-500 drop-shadow-md"
                                    }`}>
                                        {result.prediction}
                                    </p>
                                )}
                            </div>
                        </motion.div>
                    </div>
                )}

                {/* Auth Modal */}
                {showAuthModal && (
                    <div className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-50">
                        <motion.div
                            className={`w-full max-w-md p-8 rounded-2xl shadow-xl ${
                                isDarkMode
                                    ? "bg-gray-800 text-white"
                                    : "bg-white text-gray-800"
                            }`}
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ duration: 0.2 }}
                        >
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-2xl font-bold">
                                    {authMode === "login" ? "Login" : "Sign Up"}
                                </h2>
                                <button
                                    onClick={() => setShowAuthModal(false)}
                                    className={`p-2 rounded-full ${
                                        isDarkMode
                                            ? "hover:bg-gray-700"
                                            : "hover:bg-gray-200"
                                    }`}
                                >
                                    <FontAwesomeIcon icon={faTimes} />
                                </button>
                            </div>

                            {authMessage && (
                                <div className={`mb-4 p-2 rounded ${
                                    authMessage.includes("success")
                                        ? "bg-green-100 text-green-800"
                                        : "bg-red-100 text-red-800"
                                }`}>
                                    {authMessage}
                                </div>
                            )}

                            <form onSubmit={handleAuthSubmit} className="space-y-4">
                                <div>
                                    <label htmlFor="authEmail" className="block mb-1">
                                        Email
                                    </label>
                                    <input
                                        id="authEmail"
                                        type="email"
                                        value={authEmail}
                                        onChange={(e) => setAuthEmail(e.target.value)}
                                        className={`w-full p-2 rounded border ${
                                            isDarkMode
                                                ? "bg-gray-700 border-gray-600"
                                                : "bg-white border-gray-300"
                                        }`}
                                        required
                                    />
                                </div>

                                <div>
                                    <label htmlFor="authPassword" className="block mb-1">
                                        Password
                                    </label>
                                    <div className="relative">
                                        <input
                                            id="authPassword"
                                            type={showAuthPassword ? "text" : "password"}
                                            value={authPassword}
                                            onChange={(e) => setAuthPassword(e.target.value)}
                                            className={`w-full p-2 rounded border ${
                                                isDarkMode
                                                    ? "bg-gray-700 border-gray-600"
                                                    : "bg-white border-gray-300"
                                            }`}
                                            required
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowAuthPassword(!showAuthPassword)}
                                            className="absolute right-2 top-2 text-sm"
                                        >
                                            {showAuthPassword ? "Hide" : "Show"}
                                        </button>
                                    </div>
                                </div>

                                {authMode === "signup" && (
                                    <div>
                                        <label htmlFor="authConfirmPassword" className="block mb-1">
                                            Confirm Password
                                        </label>
                                        <input
                                            id="authConfirmPassword"
                                            type={showAuthPassword ? "text" : "password"}
                                            value={authConfirmPassword}
                                            onChange={(e) => setAuthConfirmPassword(e.target.value)}
                                            className={`w-full p-2 rounded border ${
                                                isDarkMode
                                                    ? "bg-gray-700 border-gray-600"
                                                    : "bg-white border-gray-300"
                                            }`}
                                            required
                                        />
                                    </div>
                                )}

                                <button
                                    type="submit"
                                    className={`w-full py-2 rounded ${
                                        isDarkMode
                                            ? "bg-blue-600 hover:bg-blue-700"
                                            : "bg-blue-500 hover:bg-blue-600 text-white"
                                    }`}
                                >
                                    {authMode === "login" ? "Login" : "Sign Up"}
                                </button>
                            </form>

                            <div className="mt-4 text-center">
                                <button
                                    onClick={toggleAuthMode}
                                    className={`text-sm ${
                                        isDarkMode
                                            ? "text-blue-400 hover:text-blue-300"
                                            : "text-blue-600 hover:text-blue-800"
                                    }`}
                                >
                                    {authMode === "login"
                                        ? "Don't have an account? Sign Up"
                                        : "Already have an account? Login"}
                                </button>
                            </div>
                        </motion.div>
                    </div>
                )}

                <div className={`w-full mt-8 p-4 rounded-lg ${
                    isDarkMode
                        ? "bg-gray-800 text-gray-300 shadow-inner shadow-indigo-950"
                        : "bg-gray-200 text-gray-800"
                }`}>
                    <p className="text-center">
                        🌿 Upload a clear photo of your plant's leaves for disease analysis. For best results, use well-lit photos showing affected areas clearly.
                    </p>
                    <p className="text-center mt-2">
                        Need help interpreting results? Visit our{' '}
                        <Link to="/diagnosis" className={`font-bold ${
                            isDarkMode ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-800"
                        }`}>
                            Diagnosis Page 🌿
                        </Link>
                    </p>
                    <p className="text-center mt-2">
                        Don't know what plants our AI software supports? Visit our{' '}
                        <Link to="/" className={`font-bold ${
                            isDarkMode ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-800"
                        }`}>
                            Home Page 🌿
                        </Link>
                    </p>
                </div>
            </motion.div>
        </div>
    );
}