import React, { useState, useEffect } from "react";
import { motion } from "framer-motion"; // Correct import
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowLeft, faArrowRight } from "@fortawesome/free-solid-svg-icons";

export default function ImageSlider() {
    const [currentIndex, setCurrentIndex] = useState(0);

    const images = [
        "480-360-max.png",
        "612-344-max.png",
        "1280-853-max.png"
    ];

    function imageLeft() {
        setCurrentIndex((prevIndex) =>
            prevIndex === 0 ? images.length - 1 : prevIndex - 1
        );
    }

    function imageRight() {
        setCurrentIndex((prevIndex) =>
            prevIndex === images.length - 1 ? 0 : prevIndex + 1
        );
    }

    useEffect(() => {
        const interval = setInterval(() => {
            imageRight();
        }, 3000);

        return () => clearInterval(interval);
    }, [currentIndex]);

    return (
        <div className="relative flex w-screen h-[100vh]">
            {/* Left Section */}
            <div className="w-1/2 h-full bg-black relative bg-gradient-to-r to-sky-800 from-lime-500 shadow-lg shadow-lime-600/50">
                <h1 className="text-white font-mono text-5xl text-left ml-12 mt-72 animate-bounce duration-2000">AI Plant Detection at Your Hands</h1>
            </div>
            
            {/* Right Section */}
            <div className="relative flex flex-col w-screen h-[100vh] bg-gray-900 items-center justify-center bg-gradient-to-r to-sky-950 from-gray-800 shadow-lg shadow-gray-500/50">
                <div className="flex items-center gap-6">
                    {/* Left Arrow */}
                    <motion.div
                        className="cursor-pointer text-white text-4xl p-4 hover:scale-110 transition"
                        onClick={imageLeft}
                    >
                        <FontAwesomeIcon icon={faArrowLeft} />
                    </motion.div>

                    {/* Image with Animation */}
                    <motion.img
                        key={currentIndex} // Forces re-render on change
                        src={`/assets/${images[currentIndex]}`}
                        alt={`Slide ${currentIndex}`}
                        className="rounded-xl max-h-[50vh] max-w-[90%] object-contain shadow-lg"
                        initial={{ opacity: 0, x: 50 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -50 }}
                        transition={{ duration: 0.5 }}
                    />

                    {/* Right Arrow */}
                    <motion.div
                        className="cursor-pointer text-white text-4xl p-4 hover:scale-110 transition"
                        onClick={imageRight}
                    >
                        <FontAwesomeIcon icon={faArrowRight} />
                    </motion.div>
                </div>
            </div>
        </div>

        
    );
}
