import { Link } from "react-router";
import { useState } from "react";
import "./Navbar.css";

export default function Navbar() {
    return (
        <div className="homepage-container">
            <div className="navbar-container">
                <nav className="navbar">
                    <div className="navbar-left">
                        <a href="/" className="logo">
                            Home
                        </a>
                    </div>
                    <div className="nav-center">
                        <ul className="nav-links">
                            <li>
                                <Link to="/upload" className="nav-button">
                                    Upload
                                </Link>
                            </li>
                            <li>
                                <Link to="/diagnosis" className="nav-button">
                                    Results & Diagnosis
                                </Link>
                            </li>
                            <li>
                                <Link to="/consultation" className="nav-button">
                                    Expert Consultation
                                </Link>
                            </li>
                            <li>
                                <Link to="/contact" className="nav-button">
                                    Contact & Support
                                </Link>
                            </li>
                        </ul>
                    </div>
                </nav>
            </div>
        </div>
    );
}
