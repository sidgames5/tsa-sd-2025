"use client";

import { motion } from "framer-motion";
import React, { useEffect, useRef, useState } from "react";
import ImageSlider from "../components/Slider.jsx";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowDown } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";
import { faPerson } from "@fortawesome/free-solid-svg-icons";

export default function App() {
    const [reviews, setReviews] = useState([]);
    const [formData, setFormData] = useState({
      name: "",
      message: "",
      profileImage: null,
    });
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isCameraAllowed, setIsCameraAllowed] = useState(false);
    const [videoStream, setVideoStream] = useState(null);
    const videoRef = useRef(null);
    const avatarList= [
        "/avatars/avatar1.png",
        "/avatars/avatar2.png",
        "/avatars/avatar3.png",
        "/avatars/avatar4.png"
    ]
  
    useEffect(() => {
      fetch("/api/get-reviews")
        .then((res) => res.json())
        .then((data) => setReviews(data.reviews || []));
    }, []);
  
    const handleChange = (e) => {
      const { name, value } = e.target;
      setFormData((prev) => ({ ...prev, [name]: value }));
    };
  
    const openModal = () => {
      setIsModalOpen(true);
      setIsCameraAllowed(false);
    };
  
    const closeModal = () => {
      setIsModalOpen(false);
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop());
        setVideoStream(null);
      }
    };
  
    const handleCameraPermission = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        setIsCameraAllowed(true);
        setVideoStream(stream);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        alert("Camera access denied or not available.");
      }
    };
  
    const handleCapture = () => {
      const canvas = document.createElement("canvas");
      const video = videoRef.current;
      if (!video) return;
  
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
  
      canvas.toBlob((blob) => {
        if (blob) {
          setFormData((prev) => ({ ...prev, profileImage: blob }));
        }
      }, "image/jpeg");
    };
  
    const handleSubmit = async (e) => {
      e.preventDefault();
      const form = new FormData();
      form.append("name", formData.name);
      form.append("message", formData.message);
      if (formData.profileImage) {
        form.append("profileImage", formData.profileImage);
      }
  
      const res = await fetch("/api/submit-review", {
        method: "POST",
        body: form,
      });
  
      const result = await res.json();
      if (result.success) {
        setReviews([result.review, ...reviews]);
        setFormData({ name: "", message: "", profileImage: null });
        closeModal();
      } else {
        alert(result.error || "Submission failed.");
      }
    };
    const [showScrollIcon, setShowScrollIcon] = useState(true);
    const [cookies] = useCookies(["darkMode"]);
    const missionRef = useRef(null);
    const isDarkMode = cookies.darkMode === true;

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
                className="flex flex-col justify-center items-center h-[100vh] shadow-lg w-full"
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


            </motion.div>

            <div className={`h-[120vh] w-full ${cookies.darkMode ? "bg-black" : "bg-gradient-to-r from-green-100 via-stone-50 to-blue-100"} `} >
                <div className="max-w-6xl mx-auto">
                    <div className="mt-20"></div>
                    {/*
                    <div className="text-center mb-12">
                        <div className="text-5xl font-bold text-yellow-500">⭐⭐⭐⭐⭐</div>
                        <h1 className="text-4xl font-extrabold text-gray-800 mt-2">4.8 STARS</h1>
                        <p className="text-gray-600 text-lg">Based on 38,792 Reviews</p>
                    </div>
                     */}
                    <h2 className={`text-4xl font-extrabold text-center mb-12 tracking-light bg-clip-text text-transparent 
                        ${isDarkMode
                            ? "bg-gradient-to-r from-blue-400 via-blue-500 to-indigo-500" 
                            : "bg-gradient-to-r from-emerald-400 via-green-500 to-lime-400"}`}>
                        Testimonials
                    </h2>
                    {/* Add Review */}
                    <div className="text-center mb-12">
                        <motion.button
                            onClick={openModal}
                            className="bg-green-600 hover:bg-green-700 text-white font-medium py-2.5 px-6 rounded-full text-lg"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            Add a Review
                        </motion.button>
                    </div>
                    {/* Modal */}
                    {/* Made it avatars rather than images */}
                    {isModalOpen && (
                        <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4 bg-opacity-60 z-50">
                            <motion.div className="bg-white p-6 rounded-2xl shadow-2xl w-full max-w-lg"
                                        initial={{ scale: 0.9, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        exit={{ scale:0.9, opacity: 0 }}
                                        drag
                            >
                                <h3 className="text-3xl font-bold mb-4 text-gray-800 text-center">Submit Your Review</h3>
                                <form onSubmit={handleSubmit} className="space-y-5">
                                <input
                                    type="text"
                                    name="name"
                                    placeholder="Your name (optional)"
                                    value={formData.name}
                                    onChange={handleChange}
                                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 text-gray-800"
                                />
                                <textarea
                                    name="message"
                                    placeholder="Write your testimonial..."
                                    value={formData.message}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-3 border border-gray-300 rounded-md h-28 resize-none focus:ring-2 focus:ring-green-500 focus:outline-none text-gray-800"
                                />

                                <div className="space-y-2">
                                    <label className="block font-medium text-gray-700">Choose an Avatar:</label>
                                    <div className="grid grid-cols-4 gap-3">
                                        {avatarList.map(( avatar, index ) => (
                                            <button
                                                key={index}
                                                type="button"
                                                onClick={() => setFormData({ ...formData, photo: avatar})}
                                                className={`border-2 rounded-xl p-1 transition ${formData.photo === avatar ? "border-green-500" : "border-transparent hover:border-gray-300"} `}
                                            >
                                                <img src={avatar} 
                                                alt={`Avatar ${index + 1}`}
                                                className="w-16 h-16 object-cover rounded-lg"
                                                />
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                {/* Avatar preview */}
                                {formData.photo && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 10}}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="flex flex-col items-center gap-2 pt-4"
                                    >
                                        <span className="text-sm text-gray-500">Selected Avatar</span>
                                        <img src={formData.photo} alt="Selected avatar"
                                            className="w-24 h-24 rounded-full border-2 border-green-500 shadow-md"
                                        />
                                    </motion.div>
                                )}
                                
                                <div className="flex justify-end space-x-4">
                                    <button
                                    type="button"
                                    onClick={closeModal}
                                    className="bg-gray-300 py-2 px-4 rounded-md hover:bg-gray-400"
                                    >
                                    Cancel
                                    </button>
                                    <button
                                    type="submit"
                                    className="bg-green-600 text-white py-2 px-6 rounded-full hover:bg-green-700"
                                    >
                                    Submit Review
                                    </button>
                                </div>
                                </form>
                            </motion.div>
                        </div>
                    )}
                    {/* All comments */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {reviews.length === 0 && (
                        <p className="text-center text-gray-500 col-span-2">No reviews yet.</p>
                        )}
                        {reviews.map((rev, idx) => (
                        <motion.div
                            key={idx}
                            className="bg-gray-50 p-6 rounded-2xl shadow-md flex flex-col items-center text-center border border-gray-100"
                            whileHover={{ scale: 1.02 }}
                        >
                            {rev.image ? (
                            <img
                                src={rev.image}
                                alt="Profile"
                                className="w-16 h-16 rounded-full object-cover mb-4 shadow"
                            />
                            ) : (
                            <div className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center mb-4">
                                <FontAwesomeIcon icon={faPerson} className="text-3xl text-gray-600" />
                            </div>
                            )}
                            <p className="italic text-gray-700">"{rev.message}"</p>
                            <p className="mt-4 font-semibold text-gray-800">{rev.name || "Anonymous"}</p>
                        </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            {/*
            <div className="p-4">
                <a href="/assets/TSA-SD Documentation-5.pdf" className={`${cookies.darkMode ? "text-white": "text-black"} underline`}>
                    Documentation Portfolio
                </a>
            </div>
            */}
        </div> 
        
    );
}
