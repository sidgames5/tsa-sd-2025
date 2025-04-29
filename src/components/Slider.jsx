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
        "corn-294-max.jpg",
        "100-486-max.jpg",
        "187-364-max.jpg",
        "bellPepper.jpg"
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
        <div className={`relative flex w-full h-[101vh] items-center justify-end align-middle bg-opacity-0 ${cookies.darkMode ? "text-white" : "text-black"}`}>
            {/* Left Section original gradient: bg-gradient-to-r to-sky-950 from-green-800*/}
            <div className={`flex flex-col items-start justify-center text-left w-1/2 h-full bg-gradient-to-l relative shadow-lg p-12 ${cookies.darkMode ? "from-black to-sky-950" : "from-white to-sky-100"}`}>
                <motion.h1
                    className={`text-6xl font-semibold ${cookies.darkMode ? "bg-gradient-to-r from-blue-400 via-blue-600 to-blue-800 bg-clip-text text-transparent" : "text-emerald-950"}`}
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}
                >
                    <span className="bg-gradient-to-r from-blue-600 to-sky-400 bg-clip-text text-transparent">LeafLogic</span>
                    <br />
                    <span className="bg-gradient-to-r from-blue-600 to-sky-400 bg-clip-text text-transparent text-4xl">
                        AI Plant Detection
                    </span>
                </motion.h1>
                {/* <motion.h3 className="text-white text-3xl ml-6 mr-6 underline cursor-pointer" initial={{ translateX: -100 }} animate={{ translateX: 0 }} transition={{ type: "spring", damping: 10, stiffness: 15 }}>Our Mission!</motion.h3> */}
                <motion.h1
                    className=" text-1xl mt-8 space-y-1"
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}
                >
                    <p>Unsure if your plant is healthy? LeafLogic makes it easy—just upload a</p> 
                    <p>photo, and our AI, trained on top crops like bell peppers, tomatoes,</p>
                    <p>and potatoes, gives you clear results. Boost your yields today!</p>
                </motion.h1>

                <motion.div className="flex flex-col items-center justify-between gap-2"
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}>
                    <div className="flex flex-row gap-2 text-5xl p-4">
                        <GiBellPepper className="text-yellow-400" />
                        <GiTomato className="text-red-400" />
                        <GiPotato className="text-orange-800" />
                    </div>
                    <motion.button
                        className=" px-6 py-3 mb-7 text-lg font-semibold rounded-lg bg-sky-600 hover:bg-sky-500 text-white shadow-md transition"
                        onClick={() => navigate("/upload")}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6, duration: 0.4 }}
                    >
                        Upload Plant Image
                    </motion.button>
                </motion.div>
            </div>

            {/* Right Section */}
            <div className={`relative flex flex-col w-1/2 h-[101vh] items-center justify-center bg-gradient-to-r ${cookies.darkMode ? "from-black to-sky-950" : "from-white to-sky-100"}`}>
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
                        key={currentIndex} // Forces re-render on change
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
