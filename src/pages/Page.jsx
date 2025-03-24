import * as motion from "motion/react-client";
import React, { useEffect, useState } from "react";
import ImageSlider from "../components/Slider.jsx";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowDown } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";


export default function App() {
    const [showScrollIcon, setShowScrollIcon] = useState(true);
    const [cookies] = useCookies(["darkMode"]);

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
        <div className="App flex flex-col justify-center items-center">
            {/* <CustomSlider>
                {images.map((image, index) => {
                    return <img key={index} src={image.imgURL} alt={image.imgAlt} />;
                })}
            </CustomSlider> */}
            <ImageSlider />
            {showScrollIcon && <motion.div
                className={`${cookies.darkMode ? "text-white hover:text-green-200" : "text-black hover:text-green-600"} fixed bottom-8 flex items-center justify-center transform -translate-x-1/2 cursor-pointer`}
                initial={{ translateY: -100 }}
                animate={{ translateY: 0 }}
                transition={{ duration: 2, type: "spring", stiffness: 100, damping: 10 }}
                onClick={() => {
                    window.scrollTo({ top: window.innerHeight, behavior: "smooth" });
                }}
            >
                <h1 className="animate-bounce mr-4 text-2xl">Our Mission</h1>
                <FontAwesomeIcon className="animate-bounce" icon={faArrowDown} fontSize={36} />
            </motion.div>}
            <motion.div
                className="info flex flex-col justify-center items-center leading-relaxes h-[100vh] w-full"
                initial={{ backgroundColor: cookies.darkMode ? "#ffffff" : "#000000" }}
                whileInView={{ backgroundColor: cookies.darkMode ? "#000000" : "#ffffff" }}
                transition={{ type: "tween", duration: 0.7 }}
            >
                <div className="w-max">
                    {/*Add Typewriter Animation*/}
                    <motion.h1
                        className="text-green-600 text-5xl font-bold drop-shadow-lg"
                        initial={{ scale: 0 }}
                        whileInView={{ scale: 1 }}
                        transition={{ type: "spring", damping: 10 }}
                    >
                        Our Mission
                    </motion.h1>
                </div>
                <table className="table-auto text-left w-[80vw] mt-20">
                    <tbody>
                        {/* About Us Secton */}
                        <tr>
                            <td
                                className={`font-bold text-4xl p-4 rounded-tl-xl ${cookies.darkMode ? 'text-white bg-sky-950' : 'text-black bg-lime-100'}`}
                            >
                                About Us
                            </td>
                            <td
                                className={`text-2xl w-[50vw] p-4 rounded-tr-xl ${cookies.darkMode ? 'text-white bg-black' : 'text-black bg-gray-100'}`}
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
                                We created this project because we wanted to blend our love for technology with a cause that can make a meaningful impact. As high schoolers, we often feel that there's a lack of real-world tech solutions made by people our age. We wanted to show that with determination and the right tools, we can create something valuable.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </motion.div>
        </div >
    );
}
