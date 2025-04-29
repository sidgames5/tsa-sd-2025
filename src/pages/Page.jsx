"use client";

import { motion } from "framer-motion";
import React, { useEffect, useRef, useState } from "react";
import ImageSlider from "../components/Slider.jsx";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowDown } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";

export default function App() {
    const [showScrollIcon, setShowScrollIcon] = useState(true);
    const [cookies] = useCookies(["darkMode"]);
    const missionRef = useRef(null);

    useEffect(() => {
        const handleScroll = () => {
            if (window.scrollY > (window.innerHeight / 4)) {
                setShowScrollIcon(false);
            } else {
                setShowScrollIcon(true);
            }
        };

        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    return (
        <div className="flex flex-col justify-center items-center">
            <ImageSlider />
            {showScrollIcon && (
                <motion.div
                    className={`${cookies.darkMode ? "text-white hover:text-sky-300" : "text-black hover:text-sky-600"} fixed bottom-8 flex items-center justify-center transform -translate-x-1/2 cursor-pointer`}
                    initial={{ translateY: 0 }}
                    animate={{ translateY: 0 }}
                    transition={{ duration: 2, type: "spring", stiffness: 100, damping: 10 }}
                    onClick={() => {
                        window.scrollTo({ top: window.innerHeight, behavior: "smooth" });
                    }}
                >
                    <h1 className="animate-bounce mr-4 text-2xl">Our Mission</h1>
                    <FontAwesomeIcon className="animate-bounce" icon={faArrowDown} fontSize={36} />
                </motion.div>
            )}
            <motion.div
                className="flex flex-col justify-center items-center h-[105vh] w-full"
                initial={{ backgroundColor: cookies.darkMode ? "#ffffff" : "#000000" }}
                whileInView={{ backgroundColor: cookies.darkMode ? "#000000" : "#ffffff" }}
                transition={{ type: "tween", duration: 0.7 }}
            >
                <div className="w-max" ref={missionRef}>
                    <motion.div
                        className="flex items-end"
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5 }}
                    >
                        {/*
                        <motion.h1
                            className={`mt-10 text-5xl font-bold drop-shadow-lg bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent ${cookies.darkMode ? 'drop-shadow-[0_0_8px_rgba(56,182,255,0.8)]' : 'drop-shadow-[0_0_8px_rgba(56,182,255,0.5)]'
                                }`}
                            initial={{ clipPath: 'inset(0 100% 0 0)' }}
                            whileInView={{ clipPath: 'inset(0 0% 0 0)' }}
                            viewport={{ once: true }}
                            transition={{
                                duration: 0.8,
                                ease: 'linear',
                            }}
                        >
                            Our Mission
                        </motion.h1>
                        */}
                        {/*
                        <motion.span
                            className={`h-12 w-1 mb-1 ml-1 ${
                                cookies.darkMode ? "bg-white" : "bg-black"
                            }`}
                            initial={{ opacity: 0 }}
                            whileInView={{ 
                                opacity: [0, 1, 0],
                            }}
                            viewport={{ once: true }}
                            transition={{
                                duration: 2,
                                repeat: Infinity,
                                repeatDelay: 0.3,
                            }}
                        /> */}
                    </motion.div>
                </div>

                <div className="px-6 py-12 space-y-16 mt-70">
                    <motion.div className="flex justify-between items-center md:flex-row item-center md:justify-between gap-8">
                        <motion.img className={`w-full md:w-1/2 max-h-[275px] rounded-2xl shadow-lg p-2 ${cookies.darkMode ? "bg-blue-950" : "bg-blue-200"}`} src="https://cdn.pixabay.com/photo/2015/06/24/15/45/student-820274_1280.jpg" 
                            alt="codeImg"
                            initial={{ opacity: 50, scale: 0.5 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.4, scale: {visualDuration: 0.8} }} 
                        />
                        <div className={`md:w-[600px] text-lg p-5 ${cookies.darkMode ? " text-white" : "text-gray-800"}`}>
                            <motion.h2 className="text-3xl mb-3" 
                                initial={{ clipPath: 'inset(0 100% 0 0)' }}
                                whileInView={{ clipPath: 'inset(0 0% 0 0)' }}
                                viewport={{ once: false }}
                                transition={{
                                    duration: 1.6,
                                    ease: 'linear',
                                }}
                            >
                                About us
                            </motion.h2>
                            <p>We are three high schoolers with a shared passion for coding, technology, and innovation. As young developers, we've spent countless hours learning and building projects that push our boundaries and expand our knowledge. We believe that coding is not just about writing lines of code, but about creating something meaningful that can make a real difference.</p>
                        </div>
                    </motion.div>
                    <motion.div className="flex justify-between items-center md:flex-row item-center md:justify-between gap-8">
                        <div className={`md:w-1/2 text-lg p-5 ${cookies.darkMode ? "text-white" : "text-gray-700"} `}>
                            <motion.h2 className="text-3xl mb-3"
                                initial={{ clipPath: 'inset(0 100% 0 0)' }}
                                whileInView={{ clipPath: 'inset(0 0% 0 0)' }}
                                viewport={{ once: false }}
                                transition={{
                                    duration: 1.6,
                                    ease: 'linear',
                                }}
                            >
                                Why we built this
                            </motion.h2>
                            <p>We created this project to address rising concerns about food shortages. With a growing global population, agriculture faces increasing pressure. Early detection of plant diseases is a key way to protect crops and boost yields. By spotting issues early, farmers can act quickly and reduce losses. Our tool offers a fast, accessible, and accurate solution to support this effort.</p>
                        </div>
                        <motion.img className={`w-full md:w-1/2 max-h-[250px] rounded-2xl shadow-lg p-2 ${cookies.darkMode ? "bg-green-950" : "bg-green-200"}`} src="https://cdn.pixabay.com/photo/2023/03/31/14/52/rice-field-7890204_1280.jpg"
                            alt="farmImg"
                            initial={{ opacity: 50, scale: 0.5 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.4, scale: {visualDuration: 0.8} }} 
                         />
                    </motion.div>
                </div>
                
                {/* 
                <table className="table-auto text-left w-[80vw] p-1 mt-10">
                    <tbody>
                        <tr>
                            <td
                                className={`font-bold text-4xl p-4 rounded-tl-xl ${cookies.darkMode ? 'text-white bg-sky-950' : 'text-black bg-sky-100'}`}
                            >
                                About Us
                            </td>
                            <td
                                className={`text-xl w-[50vw] p-4 rounded-tr-xl ${cookies.darkMode ? 'text-white bg-stone-900' : 'text-black bg-gray-100'}`}
                            >
                                We are three high schoolers with a shared passion for coding, technology, and innovation. As young developers, we've spent countless hours learning and building projects that push our boundaries and expand our knowledge. We believe that coding is not just about writing lines of code, but about creating something meaningful that can make a real difference.
                            </td>
                        </tr>
                        <tr>
                            <td
                                className={`font-bold text-4xl p-4 rounded-bl-xl ${cookies.darkMode ? 'text-white bg-sky-950' : 'text-black bg-sky-100'}`}
                            >
                                Why We Did This
                            </td>
                            <td
                                className={`text-xl w-[50vw] p-4 rounded-br-xl ${cookies.darkMode ? 'text-white bg-stone-900' : 'text-black bg-gray-100'}`}
                            >
                                We created this project in response to the growing concern over food shortages. As the global population continues to rise, the sustainability of current agricultural practices is under pressure. One of the most immediate and impactful ways to address this challenge is by increasing crop yields through early detection and prevention of plant diseases. By identifying diseases before they spread, farmers can take timely action to protect their crops, reduce losses, and improve overall productivity. Our project aims to support this goal by providing a fast, accessible, and accurate tool for detecting plant diseases.

                            </td>
                        </tr>
                    </tbody>
                </table> */}
                <div className="p-4">
                    <a href="/assets/TSA-SD Documentation-5.pdf" className={`${cookies.darkMode ? "text-white": "text-black"} underline`}>
                        Documentation Portfolio
                    </a>
                </div>

            </motion.div>
        </div> 
    );
}
