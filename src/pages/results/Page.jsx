import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import Chart from "./Chart";
import axios from "axios";

export default function ResultsPage() {
    const [cookies, setCookie, removeCookie] = useCookies(["darkMode", "user"]);
    const [userResults, setUserResults] = useState([]);
    const [loading, setLoading] = useState(true);
    const [tips] = useState([
        "Water your plants regularly but avoid overwatering",
        "Check for pests on both sides of leaves",
        "Use organic fertilizer for better plant health",
        "Prune affected leaves to prevent disease spread",
        "Ensure plants get adequate sunlight",
    ]);
    const [email, setEmail] = useState('');
    const [emailSent, setEmailSent] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [error, setError] = useState(null);

    const validateEmail = (email) => {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);

        if (!validateEmail(email)) {
            setError('Please enter a valid email address');
            return;
        }

        try {
            setIsSending(true);

            // Replace with your actual API call
            const response = await fetch('/api/send-results', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email,
                    results: plantGroups // Assuming you want to send the plant analysis result
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to send email');
            }

            setEmailSent(true);
            setTimeout(() => setEmailSent(false), 3000); // Reset after 3 seconds
        } catch (err) {
            setError(err.message || 'Failed to send email. Please try again.');
        } finally {
            setIsSending(false);
        }
    };



    useEffect(() => {
        setLoading(true);
        if (cookies.user?.email) {
            const storedResults = JSON.parse(localStorage.getItem(`plantResults_${cookies.user.email}`)) || [];
            const formattedResults = storedResults.map(result => ({
                ...result,
                date: result.date || new Date(result.id || Date.now()).toISOString().split('T')[0]
            }));
            setUserResults(formattedResults);
        } else {
            setUserResults([]);
        }
        setLoading(false);
    }, [cookies.user]);

    const groupResultsByPlant = () => {
        const plantsMap = {};
        userResults.forEach(result => {
            if (result.status === "success") {
                const resultDate = result.date || new Date(result.id || Date.now()).toISOString().split('T')[0];
                const plantName = result.prediction.includes('Healthy')
                    ? result.prediction.replace('Healthy', '').trim()
                    : result.prediction.split(' ')[0];

                if (!plantsMap[plantName]) {
                    plantsMap[plantName] = {
                        name: plantName,
                        status: result.prediction,
                        latestImage: result.image,
                        count: 1,
                        latestDate: resultDate,
                        confidence: result.confidence,
                        id: result.id
                    };
                } else {
                    plantsMap[plantName].count++;
                    if (new Date(resultDate) > new Date(plantsMap[plantName].latestDate)) {
                        plantsMap[plantName].latestImage = result.image;
                        plantsMap[plantName].latestDate = resultDate;
                        plantsMap[plantName].status = result.prediction;
                        plantsMap[plantName].confidence = result.confidence;
                        plantsMap[plantName].id = result.id;
                    }
                }
            }
        });
        return Object.values(plantsMap);
    };

    const clearHistory = () => {
        if (cookies.user?.email) {
            localStorage.removeItem(`plantResults_${cookies.user.email}`);
            setUserResults([]);
        }
    };

    const plantGroups = groupResultsByPlant();

    if (loading) {
        return (
            <div className={`flex items-center justify-center min-h-screen ${cookies.darkMode ? "bg-gray-900" : "bg-gray-100"}`}>
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    function sendEmail() { }
    const emailError = "";

    return (
        <main className={`w-full min-h-screen ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-gray-900"}`}>
            <div className="flex flex-col items-center w-full py-12 px-4">

            <h1 className="text-4xl font-bold mt-8 p-12">AI Plant Health Results</h1>

            <div className="flex flex-row w-full max-w-[80vw] gap-8">
                <div className="flex-col flex w-full gap-8">
                    {/* Left column - Chart */}
                    <div className={`flex flex-col items-center p-6 rounded-xl min-w-[60%] ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg`}>
                        <h2 className="text-2xl font-bold mb-4">Model Performance Metrics</h2>
                        <div className="h-80 min-w-full">
                            <Chart darkMode={cookies.darkMode} />
                        </div>
                        <p className="text-sm mt-2 text-center opacity-70 w-full">
                            Shows accuracy and loss metrics from recent predictions
                        </p>
                        <p className="mt-3 text-center w-full">
                            Accuracy indicates how well the model is identifying plant health correctly over time. Model's error is the loss. <br /> <br /> A lower loss typically means better performance. This graph helps track trends, improvements, or dips in the AI's reliability, especially as you upload more images.
                        </p>
                    </div>
                    {/* Tips section */}
                    <div className={`p-6 rounded-xl ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg w-full flex flex-col items-center justify-center`}>
                        <h2 className="text-2xl font-bold mb-4">Plant Care Tips</h2>
                        <div className="grid grid-cols-1 xl:grid-cols-2 min-w-full items-center justify-center align-middle">
                            {tips.map((tip, index) => (
                                <motion.div
                                    key={index}
                                    className="flex items-start gap-3 p-3 rounded-lg hover:bg-opacity-10 hover:bg-gray-500"
                                    whileHover={{ x: 5 }}
                                    transition={{ type: "spring", stiffness: 300 }}
                                >
                                    <span className={`mt-1 flex-shrink-0 ${cookies.darkMode ? "text-green-400" : "text-green-600"}`}>•</span>
                                    <span>{tip}</span>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="flex-col flex w-full min-h-full">
                    {/* Right column - Combined Plant results and Email */}
                    <div className="flex-col flex gap-8 w-full min-h-full">
                        {/* Email section */}
                        <div className={`p-6 rounded-xl ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg flex flex-col items-center justify-center *:w-full`}>
                            <h2 className="text-2xl font-bold mb-4 w-full text-center">Send Results Via Email</h2>
                            <div className="space-y-3">
                                <input
                                    type="email"
                                    placeholder="Enter your email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    className={`w-full px-4 py-2 rounded-lg border ${cookies.darkMode ? "bg-gray-700 border-gray-600 text-white placeholder-gray-400" : "bg-white border-gray-300 text-gray-900 placeholder-gray-500"} focus:outline-none focus:ring-2 focus:ring-blue-500`}
                                />
                                <button
                                    onClick={handleSubmit}
                                    disabled={isSending}
                                    className={`w-full py-2 px-4 rounded-lg font-medium text-white transition ${emailSent ? "bg-green-500 cursor-default" : "bg-blue-600 hover:bg-blue-700"}`}
                                >
                                    {emailSent ? "Email Sent!" : "Send Email"}
                                </button>
                                <div>
                                    {/* loading animation */}
                                    {isSending && (
                                        <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-blue-500 mt-2"></div>
                                    )}
                                </div>
                                {emailError && <p className="text-sm text-red-500">{emailError}</p>}
                            </div>
                        </div>

                        {/* Plant results section */}
                        <div className={`p-6 rounded-xl h-full ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg`}>
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-2xl font-bold">Your Plant Analysis History</h2>
                                {cookies.user?.email && (
                                    <button
                                        onClick={clearHistory}
                                        className={`text-sm px-4 py-2 rounded-lg font-semibold transition-colors ${cookies.darkMode
                                            ? "bg-red-600 hover:bg-red-500 text-white"
                                            : "bg-red-100 hover:bg-red-200 text-red-700"
                                            }`}
                                    >
                                        Clear History
                                    </button>
                                )}
                            </div>

                            {cookies.user?.email && (
                                <p className="text-sm mb-4 opacity-70">Logged in as: <span className="font-medium">{cookies.user.email}</span></p>
                            )}

                            {!cookies.user?.email ? (
                                <div className={`p-8 text-center rounded-lg ${cookies.darkMode ? "bg-gray-700/50" : "bg-gray-100"}`}>
                                    <p className="text-lg">Please log in to view your plant analysis history</p>
                                </div>
                            ) : plantGroups.length > 0 ? (
                                <div className="space-y-4">
                                    {plantGroups.map((plant) => (
                                        <motion.div
                                            key={plant.id}
                                            className={`p-4 rounded-lg border ${cookies.darkMode ? "border-gray-700 hover:bg-gray-700/50" : "border-gray-200 hover:bg-gray-50"} transition-all`}
                                            whileHover={{ scale: 1.01 }}
                                            transition={{ type: "spring", stiffness: 300 }}
                                        >
                                            <div className="flex items-start gap-4">
                                                {plant.latestImage && (
                                                    <img
                                                        src={plant.latestImage}
                                                        alt={plant.name}
                                                        className="w-16 h-16 object-cover rounded-lg"
                                                    />
                                                )}
                                                <div className="flex-1">
                                                    <div className="flex justify-between items-start">
                                                        <h3 className="font-bold text-lg">{plant.name}</h3>
                                                        <span className={`text-xs px-2 py-1 rounded-full ${plant.status.includes('Healthy') ? 'bg-green-500/20 text-green-600' :
                                                            'bg-red-500/20 text-red-600'
                                                            }`}>
                                                            {plant.confidence}% confidence
                                                        </span>
                                                    </div>
                                                    <p className={`text-sm ${plant.status.includes('Healthy') ? 'text-green-500' :
                                                        'text-red-500'
                                                        }`}>
                                                        {plant.status}
                                                    </p>
                                                    <div className="flex justify-between mt-1">
                                                        <p className="text-xs opacity-70">
                                                            Scans: {plant.count}
                                                        </p>
                                                        <p className="text-xs opacity-70">
                                                            Last: {plant.latestDate}
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>
                            ) : (
                                <div className={`p-8 text-center rounded-lg ${cookies.darkMode ? "bg-gray-700/50" : "bg-gray-100"}`}>
                                    <p className="text-lg">No plant analysis results found</p>
                                    <p className="text-sm mt-2 opacity-70">Upload plant images to see your analysis history here</p>
                                </div>
                            )}
                        </div>
                    </div>




                </div>
            </div>
        </div>
    </main>
    );
}