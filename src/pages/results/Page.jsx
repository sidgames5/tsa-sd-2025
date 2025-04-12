import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useCookies } from "react-cookie";
import Chart from "./Chart";

export default function ResultsPage() {
  const [cookies] = useCookies(["darkMode", "user"]);
  const [userResults, setUserResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tips] = useState([
      "Water your plants regularly but avoid overwatering",
      "Check for pests on both sides of leaves",
      "Use organic fertilizer for better plant health",
      "Prune affected leaves to prevent disease spread",
      "Ensure plants get adequate sunlight",
  ]);

  useEffect(() => {
      setLoading(true);
      if (cookies.user?.email) {
          const storedResults = JSON.parse(localStorage.getItem(`plantResults_${cookies.user.email}`)) || [];
          // Ensure all dates are properly formatted
          const formattedResults = storedResults.map(result => ({
              ...result,
              date: result.date || new Date(result.id || Date.now()).toISOString().split('T')[0]
          }));
          setUserResults(formattedResults);
      } else {
          setUserResults([]);
      }
      setLoading(false);
  }, [cookies.user]);

  const groupResultsByPlant = () => {
      const plantsMap = {};
      
      userResults.forEach(result => {
          if (result.status === "success") {
              // Ensure we have a valid date
              const resultDate = result.date || new Date(result.id || Date.now()).toISOString().split('T')[0];
              
              // Extract plant name from prediction
              const plantName = result.prediction.includes('Healthy') 
                  ? result.prediction.replace('Healthy', '').trim()
                  : result.prediction.split(' ')[0];
              
              if (!plantsMap[plantName]) {
                  plantsMap[plantName] = {
                      name: plantName,
                      status: result.prediction,
                      latestImage: result.image,
                      count: 1,
                      latestDate: resultDate,
                      confidence: result.confidence,
                      id: result.id
                  };
              } else {
                  plantsMap[plantName].count++;
                  // Keep the most recent data
                  if (new Date(resultDate) > new Date(plantsMap[plantName].latestDate)) {
                      plantsMap[plantName].latestImage = result.image;
                      plantsMap[plantName].latestDate = resultDate;
                      plantsMap[plantName].status = result.prediction;
                      plantsMap[plantName].confidence = result.confidence;
                      plantsMap[plantName].id = result.id;
                  }
              }
          }
      });
      
      return Object.values(plantsMap);
  };

  const plantGroups = groupResultsByPlant();

  if (loading) {
      return (
          <div className={`flex items-center justify-center min-h-screen ${cookies.darkMode ? "bg-gray-900" : "bg-gray-100"}`}>
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
      );
  }

  return (
      <div className={`flex flex-col items-center min-h-screen py-12 px-4 ${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-gray-900"}`}>
          <h1 className="text-4xl font-bold mt-8 p-12">AI Plant Health Results</h1>

          <div className="flex flex-col lg:flex-row w-full max-w-6xl gap-8">
              {/* Left column - Chart */}
              <div className={`flex-1 p-6 rounded-xl ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg`}>
                  <h2 className="text-2xl font-bold mb-4">Model Performance Metrics</h2>
                  <div className="h-80">
                      <Chart darkMode={cookies.darkMode} />
                  </div>
                  <p className="text-sm mt-2 text-center opacity-70">
                      Shows accuracy and loss metrics from recent predictions
                  </p>
              </div>

              {/* Right column - Plant results */}
              <div className="flex-1">
                  <div className={`p-6 rounded-xl ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg`}>
                      <h2 className="text-2xl font-bold mb-6">Your Plant Analysis History</h2>
                      
                      {!cookies.user?.email ? (
                          <div className={`p-8 text-center rounded-lg ${cookies.darkMode ? "bg-gray-700/50" : "bg-gray-100"}`}>
                              <p className="text-lg">Please log in to view your plant analysis history</p>
                          </div>
                      ) : plantGroups.length > 0 ? (
                          <div className="space-y-4">
                              {plantGroups.map((plant) => (
                                  <motion.div 
                                      key={plant.id}
                                      className={`p-4 rounded-lg border ${cookies.darkMode ? "border-gray-700 hover:bg-gray-700/50" : "border-gray-200 hover:bg-gray-50"} transition-all`}
                                      whileHover={{ scale: 1.01 }}
                                      transition={{ type: "spring", stiffness: 300 }}
                                  >
                                      <div className="flex items-start gap-4">
                                          {plant.latestImage && (
                                              <img 
                                                  src={plant.latestImage} 
                                                  alt={plant.name}
                                                  className="w-16 h-16 object-cover rounded-lg"
                                              />
                                          )}
                                          <div className="flex-1">
                                              <div className="flex justify-between items-start">
                                                  <h3 className="font-bold text-lg">{plant.name}</h3>
                                                  <span className={`text-xs px-2 py-1 rounded-full ${
                                                      plant.status.includes('Healthy') ? 'bg-green-500/20 text-green-600' : 
                                                      'bg-red-500/20 text-red-600'
                                                  }`}>
                                                      {plant.confidence}% confidence
                                                  </span>
                                              </div>
                                              <p className={`text-sm ${
                                                  plant.status.includes('Healthy') ? 'text-green-500' : 
                                                  'text-red-500'
                                              }`}>
                                                  {plant.status}
                                              </p>
                                              <div className="flex justify-between mt-1">
                                                  <p className="text-xs opacity-70">
                                                      Scans: {plant.count}
                                                  </p>
                                                  <p className="text-xs opacity-70">
                                                      Last: {plant.latestDate}
                                                  </p>
                                              </div>
                                          </div>
                                      </div>
                                  </motion.div>
                              ))}
                          </div>
                      ) : (
                          <div className={`p-8 text-center rounded-lg ${cookies.darkMode ? "bg-gray-700/50" : "bg-gray-100"}`}>
                              <p className="text-lg">No plant analysis results found</p>
                              <p className="text-sm mt-2 opacity-70">Upload plant images to see your analysis history here</p>
                          </div>
                      )}
                  </div>

                  {/* Tips section */}
                  <div className={`mt-6 p-6 rounded-xl ${cookies.darkMode ? "bg-gray-800" : "bg-white"} shadow-lg`}>
                      <h2 className="text-2xl font-bold mb-4">Plant Care Tips</h2>
                      <ul className="space-y-3">
                          {tips.map((tip, index) => (
                              <motion.li 
                                  key={index} 
                                  className="flex items-start gap-3 p-3 rounded-lg hover:bg-opacity-10 hover:bg-gray-500"
                                  whileHover={{ x: 5 }}
                                  transition={{ type: "spring", stiffness: 300 }}
                              >
                                  <span className={`mt-1 flex-shrink-0 ${cookies.darkMode ? "text-green-400" : "text-green-600"}`}>•</span>
                                  <span>{tip}</span>
                              </motion.li>
                          ))}
                      </ul>
                  </div>
              </div>
          </div>
      </div>
  );
}