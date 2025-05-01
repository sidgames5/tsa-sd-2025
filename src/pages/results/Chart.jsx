import { Line } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from "chart.js";
import { useEffect, useState } from "react";
import { useCookies } from "react-cookie";
import { getUserChartData } from './chartStuff';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const Chart = ({ darkMode }) => {
    const [cookies] = useCookies(["user"]);
    const [chartData, setChartData] = useState({
        accuracies: [],
        losses: []
    });

    // Load user-specific chart data
    useEffect(() => {
        const loadChartData = async () => {
            if (cookies.user?.email) {
                const data = getUserChartData(cookies.user.email);
                setChartData(data);
            }
        };
        loadChartData();
    }, [cookies.user]);

    // Chart styling based on theme
    const textColor = darkMode ? "#ffffff" : "#333333";
    const gridColor = darkMode ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.1)";
    const pointBackgroundColor = darkMode ? "#ffffff" : "#333333";

    return (
        <Line
            data={{
                labels: chartData.accuracies.map((_, index) => `Scan ${index + 1}`),
                datasets: [
                    {
                        label: 'Confidence',
                        data: chartData.accuracies,
                        borderColor: "#4fd1c5",
                        backgroundColor: "rgba(79, 209, 197, 0.1)",
                        borderWidth: 2,
                        tension: 0.4,
                        pointBackgroundColor,
                        pointRadius: 4,
                        yAxisID: 'y',
                    },
                    {
                        label: 'Loss',
                        data: chartData.losses,
                        borderColor: "#f687b3",
                        backgroundColor: "rgba(246, 135, 179, 0.1)",
                        borderWidth: 2,
                        tension: 0.4,
                        pointBackgroundColor,
                        pointRadius: 4,
                        yAxisID: 'y',
                    }
                ],
            }}
            options={{
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000
                },
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: textColor,
                            font: {
                                weight: 'bold',
                                size: 14
                            },
                            padding: 20,
                            usePointStyle: true,
                        }
                    },
                    tooltip: {
                        backgroundColor: darkMode ? '#2d3748' : '#ffffff',
                        titleColor: textColor,
                        bodyColor: textColor,
                        borderColor: darkMode ? '#4a5568' : '#e2e8f0',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: gridColor,
                            drawBorder: false
                        },
                        ticks: {
                            color: textColor,
                            font: {
                                weight: '500'
                            }
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 0,
                        max: 1,
                        grid: {
                            color: gridColor,
                            drawBorder: false
                        },
                        ticks: {
                            color: textColor,
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            },
                            font: {
                                weight: '500'
                            }
                        }
                    }
                }
            }}
        />
    );
};

export default Chart;