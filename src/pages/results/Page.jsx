import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useCookies } from "react-cookie";
import Chart from "./Chart";
import { getUserChartData, clearUserChartData } from "./chartStuff";
import AIChatbot from "/Users/kaniskprakash/Documents/GitHub/tsa-sd-2025/src/components/SupportAI.jsx";

export default function ResultsPage() {
    const [cookies] = useCookies(["darkMode", "user"]);
    const userEmail = cookies?.user?.email || "";
    const isDark = cookies.darkMode;
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
    const [chartData, setChartData] = useState({ accuracies: [], losses: [] });

    useEffect(() => {
        if (userEmail) {
            const storedResults = JSON.parse(localStorage.getItem(`plantResults_${userEmail}`)) || [];
            const formattedResults = storedResults.map(result => ({
                ...result,
                date: result.date || new Date(result.id || Date.now()).toISOString().split('T')[0]
            }));
            setUserResults(formattedResults);
            const initialChart = getUserChartData(userEmail);
            setChartData(initialChart);
        }
        setLoading(false);
    }, [userEmail]);

    const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        if (!validateEmail(email)) {
            setError("Please enter a valid email address");
            return;
        }
        try {
            setIsSending(true);
            const response = await fetch("/api/send-results", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email,
                    results: groupResultsByPlant(),
                }),
            });
            if (!response.ok) throw new Error("Failed to send email");
            setEmailSent(true);
            setTimeout(() => setEmailSent(false), 3000);
        } catch (err) {
            setError(err.message || "Failed to send email. Please try again.");
        } finally {
            setIsSending(false);
        }
    };

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
        if (!userEmail) return;
        localStorage.removeItem(`plantResults_${userEmail}`);
        setUserResults([]);
        const clearedChart = clearUserChartData(userEmail);
        setChartData(clearedChart);
    };

    const plantGroups = groupResultsByPlant();

    if (loading) {
        return (
            <div className={`flex items-center justify-center min-h-screen ${isDark ? "bg-gray-950" : "bg-gray-100"}`}>
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    const containerVariants = {
        hidden: { opacity: 0, y: 20 },
        show: {
            opacity: 1,
            y: 0,
            transition: {
                staggerChildren: 0.08,
                delayChildren: 0.2,
            },
        },
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 15 },
        show: { opacity: 1, y: 0 },
    };

    return (
        <main className={`relative min-h-screen w-full overflow-hidden ${isDark ? "bg-gray-950 text-white" : "bg-gradient-to-b from-gray-50 to-gray-100 text-gray-900"}`}>
            {/* Floating background lights */}
            <motion.div className="fixed inset-0 pointer-events-none z-0 ">
                <motion.div className="absolute top-1/4 left-1/4 w-64 h-64 bg-emerald-400/10 blur-3xl rounded-full" />
                <motion.div className="absolute top-[60%] right-1/4 w-96 h-96 bg-teal-400/10 blur-3xl rounded-full" />
                <motion.div className="absolute bottom-1/4 right-1/3 w-80 h-80 bg-blue-500/10 blur-3xl rounded-full" />
            </motion.div>

            <div className="relative z-10 container mx-auto px-4 py-16 max-w-7xl mt-20">
                <motion.h1
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    className="text-4xl md:text-5xl font-bold mb-12 text-center"
                >
                    Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-500">AI Plant Results</span>
                </motion.h1>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                    {/* Chart + Tips */}
                    <div className="flex flex-col gap-10">
                        {/* Chart */}
                        <motion.div
                            initial="hidden"
                            animate="show"
                            variants={containerVariants}
                            className={`rounded-xl p-6 shadow-lg border ${isDark ? "bg-gradient-to-r from-blue-900/50 to-blue-700/50 border-gray-800" : "bg-gradient-to-r from-blue-100 to-blue-200 border-gray-200"}`}
                        >
                            <motion.h2 variants={itemVariants} className="text-2xl font-bold mb-4">Model Performance</motion.h2>
                            <motion.div variants={itemVariants} className="h-80 w-full">
                                <Chart darkMode={isDark} data={chartData} />
                            </motion.div>
                            <motion.p variants={itemVariants} className="text-sm mt-4 opacity-70">
                                Confidence and loss metrics from your recent scans. Lower loss = better accuracy. Use this to track model consistency over time.
                            </motion.p>
                        </motion.div>

                        {/* Tips */}
                        <motion.div
                            initial="hidden"
                            animate="show"
                            variants={containerVariants}
                            className={`rounded-xl p-6 shadow-lg border ${isDark ? "bg-gradient-to-br from-emerald-900/50 to-emerald-700/50 border-gray-800" : "bg-gradient-to-r from-emerald-100 to-emerald-200 border-gray-200"}`}
                        >
                            <motion.h2 variants={itemVariants} className="text-2xl font-bold mb-4">Plant Care Tips</motion.h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                {tips.map((tip, i) => (
                                    <motion.div
                                        key={i}
                                        className="flex items-start gap-2"
                                        variants={itemVariants}
                                        whileHover={{ x: 5 }}
                                    >
                                        <span className="text-green-500 font-bold mt-1">•</span>
                                        <span>{tip}</span>
                                    </motion.div>
                                ))}
                            </div>
                        </motion.div>
                    </div>

                    {/* Email + History */}
                    <div className="flex flex-col gap-10">
                        {/* Email Form */}
                        <motion.div
                            className={`rounded-xl p-6 shadow-lg border ${isDark ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200"}`}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.7 }}
                        >
                            <h2 className="text-2xl font-bold mb-4 text-center">Send Results via Email</h2>
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    className={`w-full px-4 py-2 rounded-lg border ${isDark
                                        ? "bg-gray-800 border-gray-700 text-white placeholder-gray-400"
                                        : "bg-white border-gray-300 text-gray-900 placeholder-gray-500"} focus:outline-none focus:ring-2 focus:ring-emerald-400`}
                                />
                                <button
                                    type="submit"
                                    disabled={isSending}
                                    className={`w-full py-2 px-4 rounded-lg font-semibold text-white transition ${
                                        emailSent
                                            ? "bg-green-500 cursor-default"
                                            : "bg-blue-600 hover:bg-blue-700"
                                    }`}
                                >
                                    {emailSent ? "Email Sent!" : isSending ? "Sending..." : "Send Email"}
                                </button>
                                <AnimatePresence>
                                    {error && (
                                        <motion.p
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                            className="text-sm text-red-500"
                                        >
                                            {error}
                                        </motion.p>
                                    )}
                                </AnimatePresence>
                            </form>
                        </motion.div>

                        {/* History */}
                        <motion.div
                            className={`rounded-xl p-6 shadow-lg h-full border ${isDark ? "bg-gradient-to-r from-cyan-900/50 to-cyan-700/50 border-gray-800" : "bg-gradient-to-r from-cyan-100 to-cyan-200/70 border-gray-200"}`}
                            initial="hidden"
                            animate="show"
                            variants={containerVariants}
                        >
                            <div className="flex justify-between items-center mb-6">
                                    <h2 className="text-2xl font-bold">Your Plant Analysis History</h2>
                                    {userEmail && (
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

                            {userEmail ? (
                                plantGroups.length > 0 ? (
                                    <div className="space-y-4">
                                        {plantGroups.map((plant) => (
                                            <motion.div
                                                key={plant.id}
                                                className={`p-4 rounded-lg border transition ${
                                                    isDark
                                                        ? "border-gray-400 hover:bg-gray-800"
                                                        : "border-gray-400 hover:bg-gray-100"
                                                }`}
                                                whileHover={{ scale: 1.01 }}
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
                                                            <span className={`text-xs px-2 py-1 rounded-full ${plant.status.includes('Healthy') ? 'bg-green-500/20 text-green-600' : 'bg-red-500/20 text-red-600'}`}>
                                                                {plant.confidence}% confidence
                                                            </span>
                                                        </div>
                                                        <p className={`text-sm ${plant.status.includes('Healthy') ? 'text-green-500' : 'text-red-500'}`}>
                                                            {plant.status}
                                                        </p>
                                                        <div className="flex justify-between mt-1 text-xs opacity-70">
                                                            <span>Scans: {plant.count}</span>
                                                            <span>Last: {plant.latestDate}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </motion.div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-center text-sm opacity-70">No plant analysis results found. Upload images to get started.</p>
                                )
                            ) : (
                                <p className="text-center text-sm opacity-70">Please log in to view your history.</p>
                            )}
                        </motion.div>
                    </div>
                </div>
            </div>

            <AIChatbot />
        </main>
    );
}
