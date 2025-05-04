"use client"

import { Link, useLocation, useNavigate } from "react-router";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faLeaf, faMoon, faComment } from "@fortawesome/free-solid-svg-icons"
import { useCookies } from "react-cookie"
import { useState, useEffect } from "react"
import { AnimatePresence } from "motion/react"
import * as motion from "motion/react-client"

const tabs = [
    { path: "/upload", label: "Detect" },
    { path: "/diagnosis", label: "Plant Disease Guide" },
    { path: "/results", label: "AI Chart" },
    { path: "/features", label: "Features" }
]

export default function Navbar() {
    const location = useLocation()
    const navigate = useNavigate()
    const [cookies, setCookies] = useCookies(["darkMode"])
    const [selectedTab, setSelectedTab] = useState(
        tabs.find((tab) => tab.path === location.pathname) || tabs[0]
    )

    useEffect(() => {
        const activeTab = tabs.find((tab) => tab.path === location.pathname)
        if (activeTab) {
            setSelectedTab(activeTab);
        } else if (location.pathname === "/") {
            setSelectedTab({ path: "/", label: "LeafLogic" });
        }
    }, [location.pathname]);

    const isDark = cookies.darkMode

    const handleTestimonialsClick = () => {
        if (location.pathname === "/") {
            // If already on home page, scroll to testimonials
            const testimonialsSection = document.getElementById("testimonials");
            if (testimonialsSection) {
                testimonialsSection.scrollIntoView({ behavior: "smooth" });
            }
        } else {
            // If not on home page, navigate to home with hash
            navigate("/#testimonials");
            // Then scroll after the page loads
            setTimeout(() => {
                const testimonialsSection = document.getElementById("testimonials");
                if (testimonialsSection) {
                    testimonialsSection.scrollIntoView({ behavior: "smooth" });
                }
            }, 100);
        }
    }

    return (
        <div className="w-full">
            <nav
                className={`w-full p-4 flex items-center shadow-md ${isDark ? "bg-stone-950 text-white" : "bg-gray-100 text-black"
                    }`}
            >
                <div className="flex-shrink-0 flex justify-between items-center">
                    <Link
                        to="/"
                        className={`text-lg font-bold ml-8 mr-2 flex items-center bg-clip-text text-transparent ${isDark
                            ? "bg-gradient-to-r from-green-400 to-blue-500"
                            : "bg-gradient-to-r from-green-600 to-lime-500"
                            }`}
                    >
                        LeafLogic
                    </Link>
                    <FontAwesomeIcon icon={faLeaf} className="text-green-800" />
                </div>

                <div className="flex-grow flex justify-center gap-4">
                    {tabs.map((item) => (
                        <Link to={item.path} key={item.label}>
                            <motion.button
                                onClick={() => setSelectedTab(item)}
                                className={`relative px-8 py-4 rounded-md transition-colors duration-300 font-medium ${isDark
                                    ? "hover:bg-gray-700 hover: text-sky-500"
                                    : "hover:bg-green-100 text-black hover:text-green-700"
                                    }`}
                            >
                                <p>{item.label}</p>
                                {selectedTab.label === item.label && (
                                    <motion.div
                                        layoutId="underline"
                                        className={`${cookies.darkMode ? "bg-blue-500" : "bg-green-500"} absolute bottom-0 left-0 right-0 h-0.5 rounded-full`}
                                    />
                                )}
                            </motion.button>
                        </Link>
                    ))}
                    
                    {/* Testimonials Button */}
                    <motion.button
                        onClick={handleTestimonialsClick}
                        className={`relative px-8 py-4 rounded-md transition-colors duration-300 font-medium flex items-center gap-2 ${isDark
                            ? "hover:bg-gray-700 hover:text-sky-500"
                            : "hover:bg-green-100 text-black hover:text-green-700"
                            }`}
                    >
                        <FontAwesomeIcon icon={faComment} />
                        <p>Testimonials</p>
                        {location.pathname === "/" && (
                            <motion.div
                                layoutId="underline"
                                className={`${cookies.darkMode ? "bg-blue-500" : "bg-green-500"} absolute bottom-0 left-0 right-0 h-0.5 rounded-full`}
                            />
                        )}
                    </motion.button>
                </div>

                <div className="flex-shrink-0 mr-4">
                    <button
                        onClick={() => setCookies("darkMode", !cookies.darkMode)}
                        className={`p-3 rounded-full ${isDark
                            ? "hover:bg-gray-800 text-white"
                            : "hover:bg-gray-200 text-black"
                            }`}
                    >
                        <FontAwesomeIcon icon={faMoon} />
                    </button>
                </div>
            </nav>
        </div>
    )
}