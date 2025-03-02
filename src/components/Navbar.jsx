import { Link } from "react-router";
import { useState } from "react";
import { motion } from "motion/react";
import "./Navbar.css";
import NavbarButton from "./NavbarButton";
import 'font-awesome/css/font-awesome.min.css';


export default function Navbar() {
    return (
        <div className="homepage-container">
            <div className="navbar-container">
                <nav className="flex flex-row justify-between items-center bg-gray-900 text-white p-4">
                    <div className="navbar-left flex items-center">
                        <NavbarButton>
                            <a href="/" className="">Home</a>
                        </NavbarButton>
                        <i className="fa fa-leaf text-green-500 ml-2 fa-2x"></i>
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
                            <Link to="/consultation" className="nav-button">
                                Expert Consultation
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