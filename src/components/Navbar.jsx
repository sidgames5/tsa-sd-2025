import { Link } from "react-router";
import { useState } from "react";
import { motion } from "motion/react";
import "./Navbar.css";

function NavbarButton({ children }) {
    return <motion.li initial={{ scale: 1 }} whileHover={{ scale: 1.05 }} transition={{ duration: 0.3 }}>{children}</motion.li>;
}

export default function Navbar() {
    return (
        <div className="homepage-container">
            <div className="navbar-container">
                <nav className="navbar">
                    <div className="navbar-left">
                        <button className="home-btn">
                            <a href="/" className="logo">Home</a>
                        </button>
                    </div>
                    <div className="nav-center">
                        <ul className="nav-links">
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
                                <Link to="/contact" className="nav-button">
                                    Contact & Support
                                </Link>
                            </NavbarButton>
                        </ul>
                    </div>
                </nav>
            </div>
        </div>
    );
}
