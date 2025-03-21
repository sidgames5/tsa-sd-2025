import axios from "axios";
import React, { useState, useEffect } from "react";


const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await axios.post("http://127.0.0.1:5000/api/upload", formData, {
            headers: { "Content-Type": "multipart/form-data" }
        });
        console.log("Prediction:", response.data);
    } catch (error) {
        console.error("Error uploading image:", error);
    }
};

// Inside your JSX:
<input type="file" onChange={handleFileUpload} />


export default function ResultsPage() {
    // TODO: get this data from the backend
    const [plants, setPlants] = useState([
        { name: "Plant 1", status: "Healthy" },
        { name: "Plant 2", status: "Underwatered" },
        { name: "Plant 3", status: "Overwatered" },
        { name: "Plant 4", status: "Diseased" },
    ]);

    const [tips, setTips] = useState(["Water your plants regularly", "Check for pests", "Use fertilizer", "Prune dead leaves"]);

    return (
        <div className="flex flex-col items-center mt-[12vh] text-white justify-center gap-12">
            <h1 className="text-4xl font-bold">AI Results</h1>
            <div className="flex flex-row w-4/5 justify-center gap-16">
                <div className="flex flex-col w-fit h-fit border-white border-2 rounded-lg p-4 gap-2">
                    <div className="flex flex-col items-center w-full">
                        <h1 className="text-2xl text-center text-nowrap font-bold">Overall Results</h1>
                        {/* TODO: make overall health gauge */}
                    </div>
                    <div className="text-xl">
                        <h1>Tips to improve plant health</h1>
                        <ul className="list-disc list-inside text-lg">
                            {tips.map((tip, i) => <li key={i}>{tip}</li>)}
                        </ul>
                    </div>
                </div>
                <div className="flex flex-col gap-4 w-full">
                    {plants.map((plant) => <div className="flex flex-row w-full justify-between gap-8 text-xl border-white border-2 rounded-lg p-4">
                        <span>{plant.name}</span>
                        <span>{plant.status}</span>
                    </div>)}
                </div>
            </div>
        </div>
    );
}
