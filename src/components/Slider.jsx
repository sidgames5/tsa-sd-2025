import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCaretLeft, faCaretRight, faPepperHot } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";
import { useNavigate } from "react-router";
import { GiBellPepper, GiPotato, GiTomato } from "react-icons/gi";


export default function ImageSlider() {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [cookies] = useCookies(["darkMode"]);
    const navigate = useNavigate();

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
        <div className={`relative flex w-full h-screen items-center justify-center align-middle bg-opacity-0 ${cookies.darkMode ? "text-white" : "text-black"}`}>
            {/* Left Section original gradient: bg-gradient-to-r to-sky-950 from-green-800*/}
            <div className={`flex flex-col items-start justify-center text-left w-1/2 h-full bg-gradient-to-l relative shadow-lg p-12 ${cookies.darkMode ? "from-black to-sky-950" : "from-white to-sky-100"}`}>
                <motion.h1
                    className={`text-6xl font-semibold ${cookies.darkMode ? "bg-gradient-to-r from-blue-400 via-blue-600 to-blue-800 bg-clip-text text-transparent" : "text-emerald-950"}`}
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}
                >
                    <span className="bg-gradient-to-r from-blue-600 to-sky-400 bg-clip-text text-transparent">
                        LeafLogic
                    </span>
                    <br /> <span className="text-5xl bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent">AI Plant Detection</span>
                </motion.h1>

                <motion.h1
                    className=" text-lg mt-8"
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}
                >
                    Not sure if your plant is healthy? Don’t worry—LeafLogic has you covered. With an intuitive, user-friendly interface, all you need to do is upload an image on the upload page and let our AI deliver clear, actionable results. The AI is trained on high-demand crops like bell peppers, tomatoes, and potatoes. Watch your crop yields soar!
                </motion.h1>

                <motion.div className="flex flex-row gap-2 text-5xl p-4"
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}>
                    <GiBellPepper className="text-yellow-400" />
                    <GiTomato className="text-red-400" />
                    <GiPotato className="text-orange-800" />
                </motion.div>

                {/* Upload Button*/}
                <motion.button
                    className="mt-8 px-6 py-3 text-lg font-semibold rounded-lg bg-sky-600 hover:bg-sky-500 text-white shadow-md transition"
                    onClick={() => navigate("/upload")}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6, duration: 0.4 }}
                >
                    Upload Plant Image
                </motion.button>
            </div>

            {/* Right Section */}
            <div className={`relative flex flex-col w-1/2 h-[100vh] items-center justify-center bg-gradient-to-r ${cookies.darkMode ? "from-black to-sky-950" : "from-white to-sky-100"}`}>
                <div className="flex items-center gap-6 w-full justify-center">
                    {/* Left Arrow */}
                    <motion.div
                        className="cursor-pointer  text-4xl p-4 hover:scale-110 transition"
                        onClick={imageLeft}
                    >
                        <FontAwesomeIcon icon={faCaretLeft} />
                    </motion.div>

                    {/* Image with Animation */}
                    <motion.img
                        key={currentIndex}
                        src={`/assets/${images[currentIndex]}`}
                        alt={`Slide ${currentIndex}`}
                        className="rounded-xl max-h-[80vh] max-w-[75%] object-contain shadow-lg"
                        initial={{ opacity: 0, x: 50 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -50 }}
                        transition={{ duration: 0.5 }}
                    />

                    {/* Right Arrow */}
                    <motion.div
                        className="cursor-pointer text-4xl p-4 hover:scale-110 transition"
                        onClick={imageRight}
                    >
                        <FontAwesomeIcon icon={faCaretRight} />
                    </motion.div>
                </div>
            </div>
        </div>
    );
}