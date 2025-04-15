export const updateUserChartData = (userEmail, newAccuracy, newLoss) => {
    try {
        const storedData = JSON.parse(localStorage.getItem(`chartData_${userEmail}`)) || {
            accuracies: [],
            losses: []
        };
        
        const updatedData = {
            accuracies: [...storedData.accuracies, newAccuracy],
            losses: [...storedData.losses, newLoss]
        };

        localStorage.setItem(`chartData_${userEmail}`, JSON.stringify(updatedData));
        return updatedData;
    } catch (error) {
        console.error("Error updating chart data:", error);
        return {
            accuracies: [],
            losses: []
        };
    }
};

export const getUserChartData = (userEmail) => {
    try {
        return JSON.parse(localStorage.getItem(`chartData_${userEmail}`)) || {
            accuracies: [],
            losses: []
        };
    } catch (error) {
        console.error("Error getting chart data:", error);
        return {
            accuracies: [],
            losses: []
        };
    }
};

export const clearUserChartData = (userEmail) => {
    try {
        localStorage.removeItem(`chartData_${userEmail}`);
        return {
            accuracies: [],
            losses: []
        };
    } catch (error) {
        console.error("Error clearing chart data:", error);
        return {
            accuracies: [],
            losses: []
        };
    }
};

