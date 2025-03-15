import { Link } from "react-router";
import { useState } from "react";
import { motion } from "motion/react";
import "./Navbar.css";
import NavbarButton from "./NavbarButton";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faLeaf } from "@fortawesome/free-solid-svg-icons";


export default function Navbar() {
    return (
        <div className="homepage-container">
            <div className="navbar-container">
                <nav className="flex flex-row justify-between items-center bg-gray-900 text-white p-4">
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
                    </div>
                </nav>
            </div>
        </div>
    );
}