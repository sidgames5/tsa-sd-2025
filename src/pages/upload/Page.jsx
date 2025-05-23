import { useState, useRef, useEffect } from "react";
import { Link, useNavigate } from "react-router";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUpload, faSpinner, faCheck, faTimes, faUser, faCamera, faTimesCircle, faSyncAlt } from "@fortawesome/free-solid-svg-icons";
import { updateUserChartData, getUserChartData } from "../results/chartStuff";
import heic2any from 'heic2any';
import AIChatbot from "../../components/AIChatbot";


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
    const [showCameraModal, setShowCameraModal] = useState(false);
    const [stream, setStream] = useState(null);
    const [cameraError, setCameraError] = useState(null);
    const dropAreaRef = useRef(null);
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const navigate = useNavigate();
    const [plantType, setPlantType] = useState("");
    const [errorMessage, setErrorMessage] = useState(null);



    // Check if user is logged in
    const isLoggedIn = !!cookies.user;

    const constraintsRef = useRef(null);

    useEffect(() => {
        return () => {
            if (intervalId) clearInterval(intervalId);
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
            }
        };
    }, [intervalId, stream]);

    useEffect(() => {
        const checkApiStatus = async () => {
            try {
                const response = await fetch("/api/health");
                const data = await response.json();
                setApiStatus(data.status === "healthy" ? "connected" : "error");
            } catch (error) {
                setApiStatus("error");
                console.error("Backend connection error:", error);
            }
        };
        checkApiStatus();


        const interval = setInterval(checkApiStatus, 30000);
        return () => clearInterval(interval);
    }, []);

    // Camera functions
    const startCamera = async () => {
        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }

        setCameraError(null);
        try {
            // First try rear camera
            let mediaStream;
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: "environment", // Prefer rear camera
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                    },
                    audio: false,
                });
            } catch (rearError) {
                // Fallback to front camera if rear fails
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: true, // Let browser choose default (usually front) camera
                    audio: false,
                });
            }

            setStream(mediaStream);
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
            setShowCameraModal(true);
        } catch (error) {
            console.error("Camera error:", error);
            let errorMessage = "Could not access camera.";

            if (error.name === "NotAllowedError") {
                errorMessage = "Please allow camera access in your browser settings.";
            } else if (error.name === "NotFoundError") {
                errorMessage = "No camera detected on this device.";
            }

            setCameraError(errorMessage);
        }
    };
    const switchCamera = async () => {
        if (!stream) return;

        const videoTrack = stream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        const newFacingMode = settings.facingMode === "user" ? "environment" : "user";

        try {
            stopCamera();
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: newFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
                audio: false,
            });
            setStream(mediaStream);
            videoRef.current.srcObject = mediaStream;
        } catch (error) {
            console.error("Error switching camera:", error);
            setCameraError("Failed to switch camera. Please try again.");
        }
    };
    const capturePhoto = () => {
        if (!videoRef.current || !canvasRef.current) {
            setErrorMessage("Camera is not ready, so please try again.");
            // Aura?
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        try {
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (!blob) {
                    setErrorMessage("Failed to capture image. Try again.");
                    // such an auraful moment
                    return;
                }

                const file = new File([blob], `capture-${Date.now()}.jpg`, {
                    type: 'image/jpeg'
                });

                setImage(file);
                setPreview(URL.createObjectURL(blob));
                setResult(null);
                stopCamera();
                setShowCameraModal(false);
            }, 'image/jpeg', 0.9);

        } catch (error) {
            console.error("Capture error:", error);
            setErrorMessage("Failed to capture image. Please try again.");
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            setStream(null);
        }
    };

    const ErrorPopup = ({ message, onClose }) => {
        return (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
                <div className={`p-4 rounded-lg shadow-xl max-w-md w-full mx-4 ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
                    }`}>
                    <div className="flex justify-between items-center mb-3">
                        <h3 className="text-lg font-semibold">Error</h3>
                        <button
                            onClick={onClose}
                            className={`p-1 rounded-full ${isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                                }`}
                        >
                            <FontAwesomeIcon icon={faTimes} />
                        </button>
                    </div>
                    <p className="mb-4">{message}</p>
                    <button
                        onClick={onClose}
                        className={`w-full py-2 rounded ${isDarkMode ? 'bg-red-600 hover:bg-red-700' : 'bg-red-500 hover:bg-red-600 text-white'
                            }`}
                    >
                        OK
                    </button>
                </div>
            </div>
        );
    };

    const handleImageChange = async (event) => {  // Added async here cuz I got yelled at
        // for using await when it wasn't an async func so yea
        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }

        const fileInput = event.target;
        const file = fileInput.files?.[0];
        fileInput.value = null;

        if (!file) return;

        // Check for HEIC/HEIF files
        const isHeic = file.name.toLowerCase().endsWith('.heic') ||
            file.name.toLowerCase().endsWith('.heif') ||
            file.type === 'image/heic' ||
            file.type === 'image/heif';

        // Check file size first (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            alert("File is too large. Maximum size is 10MB.");
            return;
        }

        try {
            setLoading(true);  // Show loading state during processing

            let processedFile = file;
            let previewUrl;

            if (isHeic) {
                // Convert HEIC to JPEG bc I'm a sig- 
                const conversionResult = await heic2any({
                    blob: file,
                    toType: 'image/jpeg',
                    quality: 0.8
                });

                // Handle both single and multiple results cuz why not
                const resultBlob = Array.isArray(conversionResult) ? conversionResult[0] : conversionResult;
                processedFile = new File(
                    [resultBlob],
                    // chat is the replace thing aura!?!?!?
                    file.name.replace(/\.[^/.]+$/, '.jpg'),
                    { type: 'image/jpeg', lastModified: new Date().getTime() }
                );

                previewUrl = URL.createObjectURL(resultBlob);
            } else if (file.type.startsWith("image/")) {
                previewUrl = URL.createObjectURL(file);
            } else {
                throw new Error("Unsupported file format");
            }

            setImage(processedFile);
            setPreview(previewUrl);
            setResult(null);
        } catch (error) {
            console.error("Error processing image:", error);
            setErrorMessage(
                error.message.includes("unsupported")
                    ? "Unsupported file format. Please use JPEG, PNG, or HEIC."
                    : "Error processing image. Please try another file."
            );
            setPreview(null);
            setImage(null);
        } finally {
            setLoading(false);
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
        if (plantType === "Select a Plant") {
            setErrorMessage("Please select a valid plant option before uploading.");
            return;
        }

        if (!isLoggedIn) {
            setShowAuthModal(true);
            return;
        }

        if (!image) {
            setErrorMessage("Please select an image first!");
            return;
        }

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
            formData.append("plantType", plantType);

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Server error");
            }

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || "Prediction failed");
            }

            setProgress(100);
            const now = new Date();
            const newResult = {
                status: "success",
                prediction: data.prediction,
                confidence: data.confidence,
                timestamp: now.toLocaleString(),
                image: preview,
                date: now.toISOString().split("T")[0],
                id: Date.now(),
            };

            setResult(newResult);
            setErrorMessage(null);

            // Save to local storage if logged in
            if (isLoggedIn && cookies.user?.email) {
                const userEmail = cookies.user.email;
                const userResults = JSON.parse(localStorage.getItem(`plantResults_${userEmail}`)) || [];
                userResults.unshift(newResult);
                localStorage.setItem(`plantResults_${userEmail}`, JSON.stringify(userResults));
                updateUserChartData(userEmail, data.confidence / 100, 1 - (data.confidence / 100));
            }

        } catch (error) {
            console.error("Upload error:", error);
            setErrorMessage(error.message || "Analysis failed. Please try again.");
            setResult({
                status: "error",
                prediction: error.message,
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

        // Reset current image and preview
        setImage(null);
        setPreview(null);
        setResult(null);

        // Trigger file input click
        if (fileInputRef.current) {
            fileInputRef.current.value = null; // Clear any previous selection
            fileInputRef.current.click();
        }
    };

    const handleAuthSubmit = (e) => {
        e.preventDefault();

        if (
            !authEmail ||
            !authPassword ||
            (authMode === "signup" && !authConfirmPassword)
        ) {
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

            localStorage.setItem(
                authEmail,
                JSON.stringify({ password: authPassword })
            );
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
        <div
            className={`flex flex-col items-center justify-center min-h-screen w-full p-6 transition-all overflow-x-hidden overflow-y-auto scroll-smooth ${isDarkMode
                ? "bg-gradient-to-br from-gray-950 via-black to-slate-900 text-white"
                : "bg-gradient-to-br from-emerald-50 via-white to-lime-100 text-gray-900"
                }`}
        >
            <AIChatbot />
            <motion.div
                className={`p-10 max-h-[120vh] mt-20 rounded-[2.5rem] w-full max-w-6xl flex flex-col justify-center items-center border-[2px] shadow-2xl ring-1 ring-offset-2 ${isDarkMode
                    ? "bg-gradient-to-br from-neutral-950 via-sky-950 to-black border-indigo-800 shadow-indigo-900/40 ring-indigo-500/20 backdrop-blur-xl"
                    : "bg-gradient-to-br from-emerald-200 via-neutral-200 to-green-200 border-green-600 shadow-green-600/40 ring-green-600/20 backdrop-blur-md"
                    }`}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <div className="flex justify-between w-full mt-2 text-center">
                    <h1
                        className={`text-5xl font-extrabold tracking-tight mb-6 drop-shadow-xl ${isDarkMode ? "bg-gradient-to-r from-blue-200 via-teal-50 to-sky-300 text-transparent bg-clip-text" : "text-green-900"
                            }`}
                    >
                        Plant Disease Detection 🌿
                    </h1>
                    {isLoggedIn ? (
                        <button
                            onClick={handleLogout}
                            className={`flex items-center gap-2 px-4 py-2 rounded-[1.2rem] text-sm hover:scale-105 ${isDarkMode
                                ? "bg-gray-700 hover:bg-gray-600 duration-300 text-white"
                                : "bg-gray-300 hover:bg-gray-400 duration-300 text-gray-800"
                                }`}
                        >
                            <FontAwesomeIcon icon={faUser} />
                            Logout
                        </button>
                    ) : (
                        <button
                            onClick={() => setShowAuthModal(true)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm ${isDarkMode
                                ? "bg-blue-700 hover:bg-blue-600 duration-300 text-white"
                                : "bg-blue-600 hover:bg-blue-500 duration-300 text-white"
                                }`}
                        >
                            <FontAwesomeIcon icon={faUser} />
                            Login
                        </button>
                    )}
                </div>

                {apiStatus && (
                    <div
                        className={`text-sm mb-6 p-2 rounded-[1rem] ${apiStatus === "connected"
                            ? "bg-green-300 text-green-800"
                            : apiStatus === "error"
                                ? "bg-red-100 text-red-800"
                                : "bg-yellow-100 text-yellow-800"
                            }`}
                    >
                        {apiStatus === "connected"
                            ? "AI Model Connected"
                            : apiStatus === "error"
                                ? "AI Model Unavailiable - Analysis won't work"
                                : "Checking backend connection..."}
                    </div>
                )}
                {errorMessage && (
                    <div className={`fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${isDarkMode ? "bg-red-800 text-white" : "bg-red-100 text-red-800"
                        }`}>
                        <div className="flex justify-between items-center">
                            <span>{errorMessage}</span>
                            <button
                                onClick={() => setErrorMessage(null)}
                                className="ml-4 p-1 rounded-full hover:bg-black/10"
                            >
                                <FontAwesomeIcon icon={faTimes} />
                            </button>
                        </div>
                    </div>
                )}
                {!isLoggedIn && (
                    <div
                        className={`w-full p-8 mb-6 text-center rounded-lg ${isDarkMode
                            ? "bg-gray-900/80 text-gray-300"
                            : "bg-gray-100 text-gray-800"
                            }`}
                    >
                        <p className="text-lg font-medium mb-2">
                            Please login to upload and analyze plant images
                        </p>
                        <button
                            onClick={() => setShowAuthModal(true)}
                            className={`px-4 py-2 rounded-lg ${isDarkMode
                                ? "bg-blue-600 hover:bg-blue-500 text-white"
                                : "bg-blue-500 hover:bg-blue-600 text-white"
                                }`}
                        >
                            Login or Sign Up
                        </button>
                    </div>
                )}

                {isLoggedIn && !image && (
                    <div className="flex justify-between w-full gap-6 mb-8">
                        <div
                            ref={dropAreaRef}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            onClick={triggerFileInput}
                            className={`w-full p-6 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all duration-300 shadow-lg hover:shadow-2xl ring-1 ring-offset-1 ${isDarkMode
                                ? "bg-gray-800 hover:bg-gray-700 text-white border border-gray-600"
                                : "bg-white hover:bg-gray-100 text-gray-800 border border-gray-300"
                                }`}


                        >
                            <input
                                type="file"
                                ref={fileInputRef}
                                accept="image/*,.heic,.heif,.png,.jpeg,.jpg"
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
                                Supported formats: JPEG, PNG, HEIV, HEIC (max 10MB)
                            </p>
                        </div>

                        <div className="text-center text-gray-500 mb-4 mt-10">- OR -</div>

                        <button
                            onClick={startCamera}
                            className={`w-full p-6 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all duration-300 shadow-lg hover:shadow-2xl ring-1 ring-offset-1 ${isDarkMode
                                ? "bg-gray-800 hover:bg-gray-700 text-white border border-gray-600"
                                : "bg-white hover:bg-gray-100 text-gray-800 border border-gray-300"
                                }`}


                        >
                            <FontAwesomeIcon icon={faCamera} className="text-3xl text-blue-500" />
                            <span className="text-lg">Take a Photo</span>
                        </button>
                    </div>
                )}

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
                                className="w-full max-h-64 object-contain rounded-lg mx-auto h-[160px]"
                                onError={() => setPreview(null)}
                            />
                            {loading && (
                                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
                                    <div className={`text-white p-4 rounded-lg shadow-lg ${cookies.darkMode ? "bg-blue-800" : "bg-gray-700"}`}>
                                        <FontAwesomeIcon icon={faSpinner} className="animate-spin mr-2" />
                                        Processing image...
                                    </div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
                <div className="w-full flex justify-between">
                    <div className="w-1/3 mb-8 text-center ml-20">
                        <label className="block mb-3 font-semibold">
                            Select The Plant Type You Are Uploading:
                        </label>
                        <select
                            value={plantType}
                            onChange={(e) => setPlantType(e.target.value)}
                            className={`w-full p-2 border text-center rounded-xl transition ${isDarkMode
                                ? "bg-gray-700 border-white text-white"
                                : "bg-white border-gray-300"
                                }`}
                        >
                            <option value="" className="text-center">
                                Select a Plant
                            </option>
                            <option value="Apple" className="text-center">
                                Apple
                            </option>
                            <option value="Blueberry" className="text-center">
                                Blueberry
                            </option>
                            <option value="Cherry" className="text-center">
                                Cherry
                            </option>
                            <option value="Corn" className="text-center">
                                Corn
                            </option>
                            <option value="Grape" className="text-center">
                                Grape
                            </option>
                            <option value="Orange" className="text-center">
                                Orange
                            </option>
                            <option value="Peach" className="text-center">
                                Peach
                            </option>
                            <option value="Potato" className="text-center">
                                Potato
                            </option>
                            <option value="Pepper Bell" className="text-center">
                                Pepper Bell
                            </option>
                            <option value="Raspberry" className="text-center">
                                Raspberry
                            </option>
                            <option value="Soybean" className="text-center">
                                Soybean
                            </option>
                            <option value="Squash" className="text-center">
                                Squash
                            </option>
                            <option value="Strawberry" className="text-center">
                                Strawberry
                            </option>
                            <option value="Tomato" className="text-center">
                                Tomato
                            </option>
                        </select>
                    </div>
                    {isLoggedIn && (
                        <div className="flex gap-10 justify-between items-center h-1/2 mt-8 mr-16">
                            <button
                                onClick={triggerFileInput}
                                className={`px-4 py-2 rounded-lg shadow-sm transition-all ring-1 ring-offset-1 w-[180px] ${isDarkMode
                                    ? "bg-gray-800 hover:bg-gray-700 text-white ring-gray-600"
                                    : "bg-gray-200 hover:bg-gray-300 text-black ring-gray-300"
                                    }`}
                            >
                                Change Image
                            </button>
                            <button
                                onClick={handleUpload}
                                disabled={loading || !image || apiStatus !== "connected"}
                                className={`px-6 py-2 rounded-lg flex items-center gap-2 shadow-md ring-1 ring-offset-2 ${loading
                                    ? "bg-blue-600 ring-blue-700"
                                    : "bg-blue-700 hover:bg-blue-600 ring-blue-600"
                                    } text-white ${!image || apiStatus !== "connected"
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
                </div>




                <div
                    className={`w-full mt-8 p-4 rounded-lg ${isDarkMode
                        ? "bg-gray-800 text-gray-300 shadow-inner shadow-indigo-950"
                        : "bg-gray-200 text-gray-800"
                        }`}
                >
                    <p className="text-center">
                        🌿 Upload a clear photo of your plant's leaves for disease analysis.
                        For best results, use well-lit photos showing affected areas clearly.
                    </p>
                    <p className="text-center mt-2">
                        Need help interpreting results? Visit our{" "}
                        <Link
                            to="/diagnosis"
                            className={`font-bold ${isDarkMode
                                ? "text-blue-400 hover:text-blue-300"
                                : "text-blue-600 hover:text-blue-800"
                                }`}
                        >
                            Diagnosis Page 🌿
                        </Link>
                    </p>
                    <p className="text-center mt-2">
                        Don't know what plants our AI software supports? Visit our{" "}
                        <Link
                            to="/"
                            className={`font-bold ${isDarkMode
                                ? "text-blue-400 hover:text-blue-300"
                                : "text-blue-600 hover:text-blue-800"
                                }`}
                        >
                            Home Page 🌿
                        </Link>
                    </p>
                </div>
            </motion.div>

            {/* Modal for Results */}
            {result && (
                <div
                    className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-40 backdrop-blur-sm"
                    ref={constraintsRef}
                >
                    <motion.div
                        className={`w-full max-w-lg p-6 rounded-2xl shadow-2xl ring-1 ring-offset-2 transition-all duration-300 border-2
              ${result.status === "success"
                                ? isDarkMode
                                    ? "bg-gradient-to-br from-emerald-800 via-slate-800 to-emerald-600 text-emerald-100 ring-emerald-500/40"
                                    : "bg-white text-emerald-900 ring-emerald-300/60"
                                : isDarkMode
                                    ? "bg-gradient-to-br from-red-900 via-slate-800 to-rose-700 text-red-100 ring-red-400/40"
                                    : "bg-white text-red-900 ring-red-300/60"
                            }
            `}
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.25 }}
                        drag
                        dragConstraints={constraintsRef}
                    >
                        <div className="flex justify-between items-center mb-4">
                            <h2 className={`text-xl font-semibold tracking-tight ${cookies.darkMode ? "text-green-50" : "text-green-700"}`}>
                                {result.status === "success" ? "Prediction Success" : "Prediction Failed"}
                            </h2>
                            <button
                                onClick={() => setResult(null)}
                                className={`p-2 rounded-full transition ${cookies.darkMode ? "hover:bg-white/10" : "hover:bg-black/10"}`}
                            >
                                <FontAwesomeIcon className="text-xl" icon={faTimes} />
                            </button>
                        </div>

                        <div className="text-center space-y-2">
                            <p className="text-2xl font-bold tracking-tight">
                                {result.prediction}
                            </p>
                            {result.status === "success" && (
                                <>
                                    <p className={`"text-sm font-medium ${cookies.darkMode ? "text-gray-200" : "text-gray-700"}`}>
                                        Confidence: <span className="font-semibold">{result.confidence}%</span>
                                    </p>
                                    <p className={`text-xs ${cookies.darkMode ? "text-gray-200" : "text-gray-700"}`}>
                                        Timestamp: {result.timestamp}
                                    </p>
                                </>
                            )}
                        </div>
                    </motion.div>
                </div>
            )}


            {/* Auth Modal */}
            {showAuthModal && (
                <div className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-50">
                    <motion.div
                        className={`w-full max-w-md p-8 rounded-2xl shadow-xl ${isDarkMode ? "bg-gray-800 text-white" : "bg-white text-gray-800"
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
                                className={`p-2 rounded-full ${isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-200"
                                    }`}
                            >
                                <FontAwesomeIcon icon={faTimes} />
                            </button>
                        </div>

                        {authMessage && (
                            <div
                                className={`mb-4 p-2 rounded ${authMessage.includes("success")
                                    ? "bg-green-100 text-green-800"
                                    : "bg-red-100 text-red-800"
                                    }`}
                            >
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
                                    className={`w-full p-2 rounded border ${isDarkMode ? "bg-gray-700 border-gray-600" : "bg-white border-gray-300"
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
                                        className={`w-full p-2 rounded border ${isDarkMode
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
                                        className={`w-full p-2 rounded border ${isDarkMode
                                            ? "bg-gray-700 border-gray-600"
                                            : "bg-white border-gray-300"
                                            }`}
                                        required
                                    />
                                </div>
                            )}

                            <button
                                type="submit"
                                className={`w-full py-2 rounded ${isDarkMode
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
                                className={`text-sm ${isDarkMode
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

            {/* Camera Modal */}
            {showCameraModal && (
                <div className="fixed inset-0 z-50 flex justify-center items-center bg-black/80 backdrop-blur-sm px-4">
                    <motion.div
                        className={`${cookies.darkMode ? "bg-black/30" : "bg-white/10"} relative w-full max-w-2xl border border-white/20 backdrop-blur-xl rounded-3xl p-8 shadow-2xl flex flex-col items-center gap-6`}
                        initial={{ scale: 0.95, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.3, ease: 'easeOut' }}
                    >
                        {/* Switch Camera Button */}
                        <button
                            onClick={switchCamera}
                            className="absolute top-4 right-4 p-3 bg-white/10 hover:bg-white/20 text-white rounded-full backdrop-blur-md transition-all"
                            title="Switch camera"
                        >
                            <FontAwesomeIcon icon={faSyncAlt} />
                        </button>

                        {/* Camera Info */}
                        <div className="text-white/70 text-sm">
                            Using: <span className="font-medium">{stream?.getVideoTracks()[0]?.getSettings().facingMode === "environment" ? "Rear" : "Front"} Camera</span>
                        </div>

                        {/* Error State */}
                        {cameraError ? (
                            <div className="w-full text-center p-6 rounded-xl border border-red-500/30 bg-red-500/10 text-white/90">
                                <FontAwesomeIcon icon={faTimesCircle} className="text-red-400 text-5xl mb-4" />
                                <h3 className="text-2xl font-semibold mb-2">Camera Error</h3>
                                <p className="mb-4">{cameraError}</p>
                                <div className="flex justify-center gap-4">
                                    <button
                                        onClick={() => {
                                            setCameraError(null);
                                            startCamera();
                                        }}
                                        className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl shadow transition-all"
                                    >
                                        Try Again
                                    </button>
                                    <button
                                        onClick={() => {
                                            stopCamera();
                                            setShowCameraModal(false);
                                        }}
                                        className="px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl transition-all"
                                    >
                                        Close
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <>
                                {/* Camera Feed */}
                                <div className="relative w-full aspect-video rounded-xl overflow-hidden border border-white/10 shadow-md">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className="w-full h-full object-cover"
                                    />
                                    <canvas ref={canvasRef} className="hidden" />

                                    {/* Overlay UI */}
                                    <div className="absolute inset-0 flex justify-center items-center pointer-events-none">
                                        <div className="w-3/4 h-3/4 border-2 border-white/30 rounded-xl flex justify-center items-center">
                                            <span className="text-white/80 text-xs bg-black/50 px-3 py-1 rounded-full">
                                                Align plant here
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Action Buttons */}
                                <div className="flex justify-center gap-4 mt-6">
                                    <button
                                        onClick={capturePhoto}
                                        disabled={loading}
                                        className={`px-6 py-3 rounded-full font-medium flex items-center justify-center gap-2 transition shadow-lg ${loading
                                            ? "bg-gray-600 cursor-not-allowed text-white"
                                            : "bg-emerald-600 hover:bg-emerald-700 text-white"
                                            }`}
                                    >
                                        {loading ? (
                                            <>
                                                <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                                                Processing...
                                            </>
                                        ) : (
                                            <>
                                                <FontAwesomeIcon icon={faCamera} />
                                                Capture
                                            </>
                                        )}
                                    </button>
                                    <button
                                        onClick={() => {
                                            stopCamera();
                                            setShowCameraModal(false);
                                        }}
                                        className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-full shadow-lg transition"
                                    >
                                        Cancel
                                    </button>
                                </div>

                                {/* Tips */}
                                <div className="mt-4 text-white/70 text-sm text-center">
                                    <p>Make sure the plant is clearly visible within the frame.</p>
                                    <p className="mt-1">Use good lighting for the best diagnosis accuracy.</p>
                                </div>
                            </>
                        )}
                    </motion.div>
                </div>
            )}

        </div>
    );
}
