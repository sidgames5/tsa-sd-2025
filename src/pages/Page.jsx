"use client";
import { motion, AnimatePresence } from "framer-motion";
import React, { useEffect, useRef, useState } from "react";
import ImageSlider from "../components/Slider.jsx";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowDown } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";
import AIChatbot from "../components/SupportAI.jsx";


export default function App() {
   const [animationComplete, setAnimationComplete] = useState(
     sessionStorage.getItem("animationPlayed") === "true"
   );
   const [lineGrow, setLineGrow] = useState(false);
   const [lineExpand, setLineExpand] = useState(false);
  
   useEffect(() => {
     if (!animationComplete) {
       sessionStorage.setItem("animationPlayed", "true");
     }
   }, [animationComplete]);
 
   const [isLoading, setIsLoading] = useState(true);
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
   const [reviewsCleared, setReviewsCleared] = useState(false);
   const avatarList = [
       "https://cdn-icons-png.flaticon.com/512/4333/4333609.png",
       "https://cdn-icons-png.flaticon.com/512/4140/4140048.png",
       "https://cdn-icons-png.flaticon.com/512/921/921071.png",
       "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
       "https://cdn-icons-png.flaticon.com/128/2202/2202112.png",
       "https://cdn-icons-png.flaticon.com/128/1256/1256650.png",
       "https://cdn-icons-png.flaticon.com/128/15537/15537905.png",
       "https://cdn-icons-png.flaticon.com/128/11498/11498793.png",
       "https://cdn-icons-png.flaticon.com/128/16683/16683419.png",
       "https://cdn-icons-png.flaticon.com/128/3641/3641988.png",
       "https://cdn-icons-png.flaticon.com/128/2920/2920072.png",
       "https://cdn-icons-png.flaticon.com/128/3135/3135823.png",
       "https://cdn-icons-png.flaticon.com/128/4015/4015967.png",
       "https://cdn-icons-png.flaticon.com/128/4015/4015994.png",
       "https://cdn-icons-png.flaticon.com/128/484/484945.png",
       "https://cdn-icons-png.flaticon.com/128/15375/15375450.png",
       "https://cdn-icons-png.flaticon.com/128/6705/6705530.png"
   ];
   const defaultAvatar = "https://cdn-icons-png.flaticon.com/512/4333/4333609.png";
   const constraintsRef = useRef(null);
   const [isAdminModalOpen, setIsAdminModalOpen] = useState(false);
   const [adminCredentials, setAdminCredentials] = useState({ username: "", password: "" });
   const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false);
   const [showScrollIcon, setShowScrollIcon] = useState(true);
   const [cookies] = useCookies(["darkMode"]);
   const missionRef = useRef(null);
   const isDarkMode = cookies.darkMode === true;






   const handleAdminLogin = () => {
       if (adminCredentials.username === "admin" && adminCredentials.password === "LeafLogicAdmin123") {
           setIsAdminAuthenticated(true);
           setIsAdminModalOpen(false);
           setReviews([]);
           localStorage.removeItem("reviews");
           alert("All reviews have been cleared.");
       } else {
           alert("Invalid credentials.");
       }
   };


   const clearReviews = async () => {
       if (!isAdminAuthenticated) {
           setIsAdminModalOpen(true);
           return;
       }


       if (!window.confirm("Are you sure you want to clear all reviews?")) return;


       try {
           const response = await fetch("/api/admin/reviews", {
               method: "DELETE",
           });


           if (response.ok) {
               setReviews([]);
               localStorage.setItem("reviewsCleared", "true");
               localStorage.removeItem("reviews");
               alert("All testimonials cleared.");
           } else {
               alert("Failed to clear testimonials.");
           }
       } catch (error) {
           console.error("Error clearing reviews:", error);
           alert("Error clearing testimonials.");
       }
   };


   useEffect(() => {
       const reviewsClearedFlag = localStorage.getItem("reviewsCleared");
       if (reviewsClearedFlag === "true") {
           setReviews([]);
           return;
       }


       const storedReviews = localStorage.getItem("reviews");
       if (storedReviews) {
           setReviews(JSON.parse(storedReviews));
       } else {
           fetch("/api/get-reviews")
               .then((res) => res.json())
               .then((data) => {
                   if (data.reviews && data.reviews.length > 0) {
                       setReviews(data.reviews);
                       localStorage.setItem("reviews", JSON.stringify(data.reviews));
                   } else {
                       setReviews([]);
                   }
               });
       }
   }, []);


   const resetReviews = () => {
       localStorage.removeItem("reviewsCleared");
       fetch("/api/get-reviews")
           .then((res) => res.json())
           .then((data) => {
               setReviews(data.reviews || []);
               localStorage.setItem("reviews", JSON.stringify(data.reviews || []));
           })
           .catch((error) => {
               console.error("Error fetching reviews after reset:", error);
           });
   };


   const handleChange = (e) => {
       const { name, value } = e.target;
       setFormData((prev) => ({ ...prev, [name]: value }));
   };


   const handleSubmit = async (e) => {
       e.preventDefault();


       if (!formData.message.trim()) {
           alert("Please write a testimonial.");
           return;
       }


       const form = new FormData();
       form.append("name", formData.name.trim());
       form.append("message", formData.message.trim());


       if (formData.profileImage) {
           form.append("profileImage", formData.profileImage);
       } else if (formData.photo) {
           form.append("photo", formData.photo);
       }


       try {
           const res = await fetch("/api/submit-review", {
               method: "POST",
               body: form,
           });


           const result = await res.json();


           if (result.success) {
               setReviews((prev) => [result.review, ...prev]);
               localStorage.setItem("reviews", JSON.stringify([result.review, ...reviews]));
           } else {
               alert(result.error || "Submission failed.");
           }
       } catch (err) {
           console.error("Error submitting review:", err);
           alert("An error occurred during submission.");
       }


       setFormData({ name: "", message: "", photo: null, profileImage: null });
       closeModal();
   };


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


   const handleImageError = (e) => {
       e.target.src = defaultAvatar;
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


   return (
     <div className="relative flex flex-col justify-center items-center overflow-hidden">
       <AnimatePresence>
         {!animationComplete && (
           <motion.div
             className="fixed inset-0 z-50 bg-cyan-400 flex justify-center items-center"
             initial={{ opacity: 1 }}
             animate={{ opacity: 1 }} // Keep fully opaque until removal
             exit={{ opacity: 0 }} // Still keep exit animation for safety
             transition={{ duration: 0.6 }}
             key="pink-overlay"
             onAnimationComplete={() => {
               // Remove immediately when lineExpand is true
               if (lineExpand) {
                 setAnimationComplete(true);
               }
             }}
           >
             {!lineGrow && (
               <motion.div
                 className="bg-black w-[2px]"
                 initial={{ height: 0 }}
                 animate={{ height: "100vh" }}
                 transition={{
                   duration: 1.1,
                   ease: "easeInOut",
                   onComplete: () => setTimeout(() => setLineGrow(true), 200)
                 }}
               />
             )}


            {lineGrow && !lineExpand && (
              <motion.div
                className={`absolute top-0 left-1/2 h-full ${cookies.darkMode ? "bg-black" : "bg-gray-800"}`}
                initial={{ width: "2px", x: "-1px" }}
                animate={{ width: "200vw", x: "-100vw" }}
                transition={{ 
                  duration: 1, 
                  ease: "easeInOut",
                  onComplete: () => {
                    setLineExpand(true);
                    setAnimationComplete(true);
                  }
                }}
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>

    
        <AIChatbot />
        <motion.div
          className={`w-full flex flex-col items-center ${cookies.darkMode ? "bg-black" : "bg-white"}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.6 }}
        >
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
              />
            </div>
            
            <div className="px-6 py-12 space-y-16 mt-70">
              <motion.div className="flex justify-between items-center md:flex-row item-center md:justify-between gap-8">
                <motion.img className={`w-full md:w-1/2 max-h-[275px] rounded-2xl shadow-lg p-2 ${cookies.darkMode ? "bg-blue-950" : "bg-blue-200"}`} src="https://cdn.pixabay.com/photo/2015/06/24/15/45/student-820274_1280.jpg" 
                  alt="codeImg"
                  initial={{ opacity: 50, scale: 0.5 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, scale: { visualDuration: 0.8 } }} 
                />
                <div className={`md:w-[600px] text-lg p-5 ${cookies.darkMode ? " text-white" : "text-gray-800"}`}>
                  <motion.h2 className="text-3xl mb-3" 
                    initial={{ clipPath: 'inset(0 100% 0 0)' }}
                    whileInView={{ clipPath: 'inset(0 0% 0 0)' }}
                    viewport={{ once: false }}
                    transition={{ duration: 1.6, ease: 'linear' }}
                  >
                    About Us
                  </motion.h2>
                  <p>We are three high schoolers with a shared passion for coding, technology, and innovation. As young developers, we've spent countless hours learning and building projects that push our boundaries and expand our knowledge. We believe that coding is not just about writing lines of code, but about creating something meaningful that can make a real difference.</p>
                </div>
              </motion.div>
              
              <motion.div className="flex justify-between items-center md:flex-row item-center md:justify-between gap-8">
                <div className={`md:w-1/2 text-lg p-5 ${cookies.darkMode ? "text-white" : "text-gray-700"}`}>
                  <motion.h2 className="text-3xl mb-3"
                    initial={{ clipPath: 'inset(0 100% 0 0)' }}
                    whileInView={{ clipPath: 'inset(0 0% 0 0)' }}
                    viewport={{ once: false }}
                    transition={{ duration: 1.6, ease: 'linear' }}
                  >
                    Why We Built This
                  </motion.h2>
                  <p>We created this project to address rising concerns about food shortages. With a growing global population, agriculture faces increasing pressure. Early detection of plant diseases is a key way to protect crops and boost yields. By spotting issues early, farmers can act quickly and reduce losses. Our tool offers a fast, accessible, and accurate solution to support this effort.</p>
                </div>
                <motion.img className={`w-full md:w-1/2 max-h-[250px] rounded-2xl shadow-lg p-2 ${cookies.darkMode ? "bg-green-950" : "bg-green-200"}`} src="https://cdn.pixabay.com/photo/2023/03/31/14/52/rice-field-7890204_1280.jpg"
                  alt="farmImg"
                  initial={{ opacity: 50, scale: 0.5 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, scale: { visualDuration: 0.8 } }} 
                />
              </motion.div>
            </div>
          </motion.div>
          
          <div className={`h-[100vh] w-full ${cookies.darkMode ? "bg-gradient-to-r from-blue-950 via-stone-900 to-sky-950" : "bg-gradient-to-r from-green-100 via-stone-50 to-blue-100"}`}>
            <div className="max-w-6xl mx-auto">
              <div className="mt-20" />
              <motion.h2 className={`text-4xl font-extrabold text-center mb-12 tracking-light bg-clip-text text-transparent 
                ${isDarkMode
                  ? "bg-gradient-to-r from-blue-200 via-blue-200 to-indigo-200" 
                  : "bg-gradient-to-r from-emerald-400 via-green-500 to-lime-400"}`}
                initial={{ opacity: 0, y: -30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
              >
                What Do Our Users Say?
              </motion.h2>
              
              <div className="mb-12 flex justify-center items-center gap-10">
                <motion.button
                  onClick={openModal}
                  className={`${cookies.darkMode ? "bg-blue-600 hover:bg-blue-700" : "bg-green-600 hover:bg-green-700"} text-white font-medium py-2.5 px-6 rounded-full text-lg`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Add a Review
                </motion.button>
                <button
                  onClick={clearReviews}
                  className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-6 rounded-full"
                >
                  Clear Testimonial History
                </button>
              </div>
              
              {isModalOpen && (
                <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4 bg-opacity-60 z-50" ref={constraintsRef}>
                  <motion.div className="bg-white p-6 rounded-2xl shadow-2xl w-full max-w-2xl h-fit max-h-[85vh] overflow-y-auto"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale:0.9, opacity: 0 }}
                    drag
                    dragConstraints={constraintsRef}
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
                          <div className="col-span-4">
                            <label className="block mb-2 text-sm font-medium text-gray-700">Choose an Avatar</label>
                            <div className="grid grid-cols-7 gap-4 mb-4">
                              {avatarList.map((avatar, index) => (
                                <button
                                  key={index}
                                  type="button"
                                  className={`rounded-full overflow-hidden border-4 ${
                                    formData.photo === avatar ? "border-green-500" : "border-transparent"
                                  }`}
                                  onClick={() => setFormData({ ...formData, photo: avatar })}
                                >
                                  <img 
                                    src={avatar}
                                    alt={`Avatar ${index + 1}`} 
                                    className="w-16 h-16 object-cover" 
                                    onError={handleImageError}
                                  />
                                </button>
                              ))}
                            </div>
                          </div>
                          
                          {formData.photo && (
                            <motion.div
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="flex flex-col items-center gap-2 pt-4 col-span-4"
                            >
                              <span className="text-sm text-gray-500">Selected Avatar</span>
                              <img
                                src={formData.photo}
                                alt="Selected avatar"
                                className="w-24 h-24 rounded-full border-2 border-green-500 shadow-md"
                                onError={handleImageError}
                              />
                            </motion.div>
                          )}
                        </div>
                      </div>
                      
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
              
              {isAdminModalOpen && (
                <div className="fixed inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4 z-50">
                  <motion.div
                    className="bg-white p-6 rounded-xl shadow-xl w-full max-w-md"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                  >
                    <h3 className="text-xl font-bold mb-4 text-gray-800 text-center">Admin Login</h3>
                    <input
                      type="text"
                      name="username"
                      placeholder="Username"
                      value={adminCredentials.username}
                      onChange={(e) =>
                        setAdminCredentials({ ...adminCredentials, username: e.target.value })
                      }
                      className="w-full mb-3 p-3 border border-gray-300 rounded-md text-gray-800"
                    />
                    <input
                      type="password"
                      name="password"
                      placeholder="Password"
                      value={adminCredentials.password}
                      onChange={(e) =>
                        setAdminCredentials({ ...adminCredentials, password: e.target.value })
                      }
                      className="w-full mb-4 p-3 border border-gray-300 rounded-md text-gray-800"
                    />
                    <div className="flex justify-between">
                      <button
                        onClick={() => setIsAdminModalOpen(false)}
                        className="bg-gray-300 px-4 py-2 rounded hover:bg-gray-400"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleAdminLogin}
                        className="bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700"
                      >
                        Log In
                      </button>
                    </div>
                  </motion.div>
                </div>
              )}
              
              <div className="max-h-[500px] overflow-y-auto px-4">
                {reviews.length === 0 ? (
                  <p className={`${cookies.darkMode ? "text-white" : "text-gray-500"} text-center`}>No reviews yet</p>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 justify-center mt-2">
                    {reviews.map((review, index) => (
                      <motion.div
                        key={index}
                        className={`${cookies.darkMode ? "bg-gradient-to-br from-black via-stone-950 to-black text-white" : "bg-stone-100 text-black"} shadow-xl rounded-xl p-6 w-full max-w-xl border-t-4 ${
                          index % 3 === 0
                            ? "border-green-400"
                            : index % 3 === 1
                            ? "border-blue-400"
                            : "border-yellow-400"
                        }`}
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.4, delay: index * 0.1 }}
                        whileHover={{ scale: 1.04 }}
                      >
                        <div className="flex items-center gap-4 mb-4">
                          <img
                            src={review.image || defaultAvatar}
                            alt="User avatar"
                            className="w-14 h-14 rounded-full object-cover border border-gray-300"
                          />
                          <h3 className={`${cookies.darkMode ? "text-white" : "text-gray-800"} text-lg font-semibold`}>
                            {review.name || "Anonymous"}
                          </h3>
                        </div>
                        <p className={`${cookies.darkMode ? "text-white" : "text-gray-700"} text-base break-words whitespace-pre-wrap`}>{review.message}</p>
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
}
