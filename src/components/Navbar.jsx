import { Link } from "react-router";
import { useState } from "react";
import { motion } from "motion/react";
import "./Navbar.css";
import NavbarButton from "./NavbarButton";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faLeaf, faMoon } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";


export default function Navbar() {
    const [cookies, setCookies] = useCookies(["darkMode"]);

    return (
        <div className="homepage-container">
            <div className="navbar-container">
                <nav className={`flex flex-row justify-between items-center ${cookies.darkMode ? "bg-gray-900 text-gray-100" : "bg-white text-black"} p-4`}>
                    <div className="navbar-left flex items-center">
                        <NavbarButton>
                            <div className="flex flex-row gap-1 items-center">
                                <a href="/" className="">Home</a>
                                <FontAwesomeIcon icon={faLeaf} />
                            </div>
                        </NavbarButton>
                    </div>
                    <div className="flex flex-row gap-2">
                        <NavbarButton>
                            <Link to="/upload" className="nav-button">
                                Upload
                            </Link>
                        </NavbarButton>
                        <NavbarButton>
                            <Link to="/diagnosis" className="nav-button">
                                Results & Diagnosis
                            </Link>
                        </NavbarButton>
                        <NavbarButton>
                            <Link to="/results" className="nav-button">
                                AI Results
                            </Link>
                        </NavbarButton>
                        <NavbarButton>
                            <Link to="/features" className="nav-button">
                                Features
                            </Link>
                        </NavbarButton>
                        <NavbarButton>
                            <button onClick={() => {
                                if (cookies.darkMode) {
                                    setCookies("darkMode", false);
                                } else {
                                    setCookies("darkMode", true);
                                }
                            }}>
                                <FontAwesomeIcon icon={faMoon} />
                            </button>
                        </NavbarButton>
                    </div>
                </nav>
            </div>
        </div>
    );
}