import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";
import { useEffect, useState } from "react";
import axios from "axios";

// dont touch this, everything will break
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function Chart() {
    const [data, setData] = useState([64.74, 82.95, 87.15]);

    useEffect(() => {
        async function fetchData() {
            try {
                //TODO: replace this with correct URL
                const response = await axios.get("/api/accuracy");
                if (response.data) {
                    setData(response.data);
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
                labels: [1, 2, 3],
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