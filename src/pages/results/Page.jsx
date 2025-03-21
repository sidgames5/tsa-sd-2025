import axios from "axios";
import React, { useState, useEffect } from "react";
import { motion } from "motion/react";

export default function ResultsPage() {
    // State for plant health results
    const [plants, setPlants] = useState([
        { name: "Plant 1", status: "Healthy" },
        { name: "Plant 2", status: "Unknown" },
        { name: "Plant 3", status: "Unknown" },
        { name: "Plant 4", status: "Unknown" },
    ]);

    const [tips, setTips] = useState([
        "Water your plants regularly",
        "Check for pests",
        "Use fertilizer",
        "Prune dead leaves",
    ]);

    const [accuracyData, setAccuracyData] = useState([]); // Store accuracy over epochs
    const [chartUrl, setChartUrl] = useState(""); // Store accuracy chart URL

    // Fetch training accuracy from backend
    useEffect(() => {
        const fetchAccuracyData = async () => {
            try {
                await axios.get("/api/train");
                const response = await axios.get("/api/accuracy/data");
                setAccuracyData(response.data.accuracy); // Store accuracy array
            } catch (error) {
                console.error("Error fetching accuracy data:", error);
            }
        };
        fetchAccuracyData();
    }, []);

    return (
        <div className="flex flex-col items-center mt-[12vh] text-white justify-center gap-12">
            <h1 className="text-4xl font-bold mt-12">AI Plant Health Results</h1>

            {/* Overall Results & Accuracy Graph */}
            <motion.div className="flex flex-row w-4/5 justify-center gap-16 cursor-pointer">
                <motion.div className="flex flex-col w-1/2 h-2/3 border-white border-2 rounded-lg p-4 gap-2">
                    <motion.li className="flex flex-col items-center w-full"
                        whileHover={{ scale: 1.15 }}
                        transition={{ type: "spring", stiffness: 50 }}
                    >
                        <h1 className="text-2xl text-center text-nowrap font-bold">Training Accuracy</h1>
                        {chartUrl && <img src={chartUrl} alt="Accuracy Chart" className="mt-2 w-full max-w-xs" />}
                    </motion.li>
                    <div className="text-xl">
                        <h1>Tips to improve plant health</h1>
                        <ul className="list-disc list-inside text-lg">
                            {tips.map((tip, i) => <li key={i}>{tip}</li>)}
                        </ul>
                    </div>
                </motion.div>

                {/* Plant Health Status List */}
                <div className="flex flex-col gap-4 w-full">
                    {plants.map((plant, index) => (
                        <div key={index} className="flex flex-row w-full justify-between gap-8 text-xl border-white border-2 rounded-lg p-4">
                            <span>{plant.name}</span>
                            <span>{plant.status}</span>
                        </div>
                    ))}
                </div>
            </motion.div>
        </div>
    );
}
