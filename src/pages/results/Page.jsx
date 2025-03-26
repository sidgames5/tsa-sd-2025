import axios from "axios";
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";  // Changed from "motion/react"
import { useCookies } from "react-cookie";
import Chart from "./Chart";

export default function ResultsPage() {
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

    const [accuracyData, setAccuracyData] = useState({
        epochs: [],
        values: []
    });

    const [cookies] = useCookies(["darkMode"]);

    // Fetch training accuracy from backend
    useEffect(() => {
        const fetchAccuracyData = async () => {
            try {
                await axios.get("/api/train");
                const response = await axios.get("/api/accuracy/data");
                setAccuracyData({
                    epochs: response.data.epochs,
                    values: response.data.accuracy
                });
            } catch (error) {
                console.error("Error fetching accuracy data:", error);
            }
        };
        fetchAccuracyData();
    }, []);

    return (
        <div className={`flex flex-col items-center h-full justify-center gap-12 ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
            <h1 className="text-4xl font-bold">AI Plant Health Results</h1>

            {/* Overall Results & Accuracy Graph */}
            <div className="flex flex-row w-4/5 justify-center gap-16 cursor-pointer">
                <div className={`flex flex-col w-1/2 h-2/3 ${cookies.darkMode ? "border-white" : "border-black"} border-2 rounded-lg p-4 gap-2`}>
                    <li className="flex flex-col items-center w-full">
                        <h1 className="text-2xl text-center text-nowrap font-bold">Training Accuracy</h1>
                        {accuracyData.values.length > 0 && (
                            <Chart epochs={accuracyData.epochs} accuracy={accuracyData.values} />
                        )}
                    </li>
                </div>

                {/* Plant Health Status List */}
                <div className="flex flex-col gap-4 w-full">
                    {plants.map((plant, index) => (
                        <motion.div 
                            key={index} 
                            className={`flex flex-row w-full justify-between gap-8 text-xl ${cookies.darkMode ? " border-white" : "border-black"} border-2 rounded-lg p-4`}
                            whileHover={{ scale: 1.05 }}
                            transition={{ type: "spring", stiffness: 50 }}
                        >
                            <span>{plant.name}</span>
                            <span>{plant.status}</span>
                        </motion.div>
                    ))}
                </div>
            </div>
        </div>
    );
}
