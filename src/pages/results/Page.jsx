import axios from "axios";
import React, { useState, useEffect } from "react";

export default function ResultsPage() {
    const [chartUrl, setChartUrl] = useState("");

    useEffect(() => {
        axios.get("/api/accuracy-chart", { responseType: "blob" })
            .then((response) => {
                const url = URL.createObjectURL(response.data);
                setChartUrl(url);
            })
            .catch((error) => {
                console.error("Error fetching chart:", error);
            });
    }, []);

    return (
        <div className="flex flex-col items-center">
            <h1 className="text-2xl font-bold mb-4">Training Accuracy</h1>
            {chartUrl ? (
                <img src={chartUrl} alt="Accuracy Chart" className="rounded-lg shadow-lg" />
            ) : (
                <p>Loading accuracy chart...</p>
            )}
        </div>
    );
}
