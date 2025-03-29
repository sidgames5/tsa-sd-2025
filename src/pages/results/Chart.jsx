import { Line } from "react-chartjs-2";
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend 
} from "chart.js";
import { useEffect, useState } from "react";
import axios from "axios";

ChartJS.register(
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  Title, 
  Tooltip, 
  Legend
);

export default function Chart({ onDataLoaded }) {
  const [chartData, setChartData] = useState({
    accuracies: [],
    losses: []
  });

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await axios.get("/api/accuracy/chart");
        if (response.data && response.data.data) {
          setChartData(response.data.data);
          onDataLoaded(true); // Notify parent component data exists
        } else {
          onDataLoaded(false); // No data available
        }
      } catch (error) {
        console.error("Error fetching chart data:", error);
        onDataLoaded(false); // Error occurred
      }
    }
    fetchData();
  }, [onDataLoaded]);

  const data = {
    labels: chartData.accuracies.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Accuracy',
        data: chartData.accuracies,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        yAxisID: 'y',
      },
      {
        label: 'Loss',
        data: chartData.losses,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        yAxisID: 'y1',
      }
    ]
  };

  return chartData.accuracies.length > 0 ? (
    <Line data={data} />
  ) : null;
}