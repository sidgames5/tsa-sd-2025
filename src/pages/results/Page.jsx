import React, { useState } from "react";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import Chart from "./Chart";

export default function ResultsPage() {
  const [plants] = useState([
    { name: "Plant 1", status: "Healthy" },
    { name: "Plant 2", status: "Unknown" },
    { name: "Plant 3", status: "Unknown" },
    { name: "Plant 4", status: "Unknown" },
  ]);

  const [tips] = useState([
    "Water your plants regularly",
    "Check for pests",
    "Use fertilizer",
    "Prune dead leaves",
  ]);

  const [cookies] = useCookies(["darkMode"]);
  const [hasData, setHasData] = useState(false); // Track if chart data exists

  return (
    <div className={`flex flex-col items-center h-full justify-center gap-12 ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"}`}>
      <h1 className="text-4xl font-bold">AI Plant Health Results</h1>

      <div className="flex flex-row w-4/5 justify-center gap-16 cursor-pointer">
        <div className={`flex flex-col w-fit h-fit ${cookies.darkMode ? "border-white" : "border-black"} border-2 rounded-lg p-8 gap-2`}>
          <div className="flex flex-col items-center w-full">
            <h1 className="text-2xl text-center text-nowrap font-bold">Training Metrics</h1>
            <div className="flex justify-center items-center w-full h-fit mt-4">
              <Chart onDataLoaded={setHasData} />
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4 w-full">
          {plants.map((plant, index) => (
            <motion.div 
              key={index} 
              className={`flex flex-row w-full justify-between gap-8 text-xl ${cookies.darkMode ? "border-white" : "border-black"} border-2 rounded-lg p-4`}
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 50 }}
            >
              <span>{plant.name}</span>
              <span>{plant.status}</span>
            </motion.div>
          ))}
        </div>
        
        {!hasData && (
          <div className="text-yellow-500">
            {/* No training data available. Run training first. */}

          </div>
        )}
      </div>
    </div>
  );
}