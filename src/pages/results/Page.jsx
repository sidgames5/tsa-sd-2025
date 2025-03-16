import axios from "axios";
import * as motion from "motion/react-client";
import React from "react";
import { useState, useEffect } from "react";


export default function ResultsPage() {
    const [chartUrl, setChartUrl] = useState("");

    useEffect(() => {
        // fetch("/accuracy-chart")
        //     .then((response) => {
        //         if (response.ok) return response.blob();
        //         throw new Error("Failed to fetch chart");
        //     })
        //     .then((blob) => {
        //         setChartUrl(URL.createObjectURL(blob));
        //     })
        //     .catch((error) => console.error(error));
        axios.get("/accuracy-chart", { responseType: "blob" })
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
