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
                    className={`${cookies.darkMode ? "text-white hover:text-sky-450" : "text-black hover:text-sky-600"} fixed bottom-8 flex items-center justify-center transform -translate-x-1/2 cursor-pointer`}
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
                className="flex flex-col justify-center items-center h-[100vh] w-full"
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
                        <motion.h1
                            className={`text-5xl font-bold drop-shadow-lg bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent ${
                                cookies.darkMode ? 'drop-shadow-[0_0_8px_rgba(56,182,255,0.6)]' : ''
                            }`}
                            initial={{ width: 0 }}
                            whileInView={{ width: "100%" }}
                            viewport={{ once: true }}
                            transition={{
                                duration: 1.5,
                                ease: [0.04, 0.62, 0.23, 0.98],
                            }}
                        >
                            Our Mission
                        </motion.h1>
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
                                duration: 0.8,
                                repeat: Infinity,
                                repeatDelay: 0.2,
                                delay: 1.5,
                            }}
                        />
                    </motion.div>
                </div>
                
                <table className="table-auto text-left w-[80vw] mt-20">
                    <tbody>
                        <tr>
                            <td
                                className={`font-bold text-4xl p-4 rounded-tl-xl ${cookies.darkMode ? 'text-white bg-sky-950' : 'text-black bg-lime-100'}`}
                            >
                                About Us
                            </td>
                            <td
                                className={`text-2xl w-[50vw] p-4 rounded-tr-xl ${cookies.darkMode ? 'text-white bg-neutral-900' : 'text-black bg-gray-100'}`}
                            >
                                We are three high schoolers with a shared passion for coding, technology, and innovation. As young developers, we've spent countless hours learning and building projects that push our boundaries and expand our knowledge. We believe that coding is not just about writing lines of code, but about creating something meaningful that can make a real difference.
                            </td>
                        </tr>
                        <tr>
                            <td
                                className={`font-bold text-4xl p-4 rounded-bl-xl ${cookies.darkMode ? 'text-white bg-sky-950' : 'text-black bg-lime-100'}`}
                            >
                                Why We Did This
                            </td>
                            <td
                                className={`text-2xl w-[50vw] p-4 rounded-br-xl ${cookies.darkMode ? 'text-white bg-black' : 'text-black bg-gray-100'}`}
                            >
                                We created this project because crops have been growing infected daily, and we wanted to provide farmers a way to identify them and provide them with solutions on how to treat their crops that are diseases. With our free solution that can be accessed anywhere, we wanted to show that with determination and the right tools, a valuable product can be created, which helps the most important people in the United States, which are farmers.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </motion.div>
        </div>
    );
}