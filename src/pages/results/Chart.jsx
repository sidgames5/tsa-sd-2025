import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";
import { useState } from "react";

// dont touch this, everything will break
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function Chart() {
    const [data, setData] = useState([64.74, 82.95, 87.15]);

    //TODO: load the data from the server

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