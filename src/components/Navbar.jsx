"use client"

import { Link, useLocation } from "react-router";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faLeaf, faMoon } from "@fortawesome/free-solid-svg-icons"
import { useCookies } from "react-cookie"
import { useState, useEffect } from "react"
import { AnimatePresence } from "motion/react"
import * as motion from "motion/react-client"

const tabs = [
    { path: "/upload", label: "Upload" },
    { path: "/diagnosis", label: "Diagnosis" },
    { path: "/results", label: "AI Chart" },
    { path: "/features", label: "Features" }
]

export default function Navbar() {
    const location = useLocation()
    const [cookies, setCookies] = useCookies(["darkMode"])
    const [selectedTab, setSelectedTab] = useState(
        tabs.find((tab) => tab.path === location.pathname) || tabs[0]
    )

    useEffect(() => {
        const activeTab = tabs.find((tab) => tab.path === location.pathname)
        if (activeTab) setSelectedTab(activeTab)
    }, [location.pathname])

    const isDark = cookies.darkMode

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
                        LeafLogic 3.0
                    </Link>
                    <FontAwesomeIcon icon={faLeaf} className="text-green-800" />
                </div>

                <div className="flex-grow flex justify-center gap-4">
                    {tabs.map((item) => (
                        <Link to={item.path}>
                            <motion.button
                                key={item.label}
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
