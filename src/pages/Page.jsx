import * as motion from "motion/react-client";
import React, { useEffect, useState } from "react";
import "../components/CustomSlider"
import ImageSlider from "../components/Slider.jsx";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowDown } from "@fortawesome/free-solid-svg-icons";

const images = [
    {
        imgURL:
            "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-1"
    },
    {
        imgURL:
            "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-2"
    },
    {
        imgURL:
            "https://images.pexels.com/photos/1128678/pexels-photo-1128678.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-3"
    },
    {
        imgURL:
            "https://images.pexels.com/photos/54455/cook-food-kitchen-eat-54455.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-4"
    }
];

export default function App() {
    const [showScrollIcon, setShowScrollIcon] = useState(true);

    useEffect(() => {
        const handleScroll = () => {
            if (window.scrollY > window.innerHeight) {
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
            {showScrollIcon && <motion.div className="text-white fixed bottom-8 left-1/2 transform -translate-x-1/2 cursor-pointer"
                initial={{ translateY: -100 }}
                animate={{ translateY: 0 }}
                transition={{ duration: 2, type: "spring", stiffness: 100, damping: 10 }}
                onClick={() => {
                    window.scrollTo({ top: window.innerHeight, behavior: "smooth" });
                }}>
                <FontAwesomeIcon className="animate-bounce" icon={faArrowDown} fontSize={36} />
            </motion.div>}
            <motion.div className="info flex flex-col justify-center items-center leading-relaxes h-[100vh] w-full" initial={{ backgroundColor: "#082f49" }} whileInView={{ backgroundColor: "#ffffff" }} transition={{ delay: 0.25, type: "spring", stiffness: 100 }}>
                <div className="w-max">
                    {/*Add Typewriter Animation*/}
                    <motion.h1 className="text-green-600 text-5xl font-bold drop-shadow-lg"
                        initial={{ scale: 0 }} whileInView={{ scale: 1 }} transition={{ type: "spring", damping: 10 }}>Our Mission</motion.h1>
                </div>
                <table className="table-auto text-left w-[80vw] mt-20">
                    <tbody>
                        {/* About Us Secton */}
                        <tr>
                            <td className="font-bold text-4xl p-4 bg-lime-50 rounded-tl-xl border-l-4">About Us</td>
                            <td className="text-2xl w-[50vw] p-4 bg-gray-100 rounded-tr-xl border-r-4">
                                We are three high schoolers with a shared passion for coding, technology, and innovation. As young developers, we've spent countless hours learning and building projects that push our boundaries and expand our knowledge. We believe that coding is not just about writing lines of code, but about creating something meaningful that can make a real difference.
                            </td>
                        </tr>
                        <tr>
                            <td className="font-bold text-4xl p-4 bg-lime-50 rounded-bl-xl border-l-4">Why We Did This</td>
                            <td className="text-2xl w-[50vw] p-4 bg-gray-100 ounded-br-xl border-r-4">
                                We created this project because we wanted to blend our love for technology with a cause that can make a meaningful impact. As high schoolers, we often feel that there’s a lack of real-world tech solutions made by people our age. We wanted to show that with determination and the right tools, we can create something valuable.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </motion.div>
        </div >
    );
}
