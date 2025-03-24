import React, { useState, useEffect } from "react";
import { motion } from "framer-motion"; // Correct import
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCaretLeft, faCaretRight } from "@fortawesome/free-solid-svg-icons";
import { Link } from "react-router";
import { useCookies } from "react-cookie";


export default function ImageSlider() {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [cookies] = useCookies(["darkMode"]);

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
        <div className={`relative flex w-full h-[101vh] items-center justify-center align-middle bg-opacity-0 ${cookies.darkMode ? "text-white" : "text-black"}`}>
            {/* Left Section original gradient: bg-gradient-to-r to-sky-950 from-green-800*/}
            <div className={`flex flex-col items-start justify-center text-left w-1/2 h-full bg-gradient-to-l relative shadow-lg p-12 ${cookies.darkMode ? "from-black to-sky-950" : "from-white to-sky-100"}`}>
                <motion.h1
                    className=" text-6xl font-semibold"
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}
                >
                    AI Plant Detection <br /> <span className="text-5xl">at Your Hands</span>
                </motion.h1>
                {/* <motion.h3 className="text-white text-3xl ml-6 mr-6 underline cursor-pointer" initial={{ translateX: -100 }} animate={{ translateX: 0 }} transition={{ type: "spring", damping: 10, stiffness: 15 }}>Our Mission!</motion.h3> */}
                <motion.h1
                    className=" text-lg mt-8"
                    initial={{ translateX: -100 }}
                    animate={{ translateX: 0 }}
                    transition={{ type: "spring", damping: 10, stiffness: 15 }}
                >
                    Insanely easy User Interface. Simply upload an image and watch as AI gives you easy to understand results. Apply these tips and watch as your crop yields begin to increase. <span className="text-green-300 cursor-pointer underline"><Link to="/upload">Upload Plant Image Now!</Link></span>
                </motion.h1>
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
