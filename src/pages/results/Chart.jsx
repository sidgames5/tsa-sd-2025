import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";
import { useEffect, useState } from "react";
import axios from "axios";

// dont touch this, everything will break
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function Chart() {
    const [data, setData] = useState([]);

    useEffect(() => {
        async function fetchData() {
            try {
                //TODO: replace this with correct URL
                const response = await axios.get("/api/accuracy/chart");
                if (response.data) {
                    setData(response.data.data.accuracies);
                }
            } catch (error) {
                console.error(error);
            }
        }
        fetchData();
    }, []);

    return (
        <Line
            datasetIdKey='id'
            data={{
                labels: Array.from({length: data.length}, (_, i) => i + 1),
                datasets: [
                    {
                        id: 1,
                        label: 'Accuracy',
                        data: data,
                        backgroundColor: "#ffffff",
                        borderColor: "#ffffff",
                        tension: 0.3
                    }
                ],
            }}
        />
    );
}