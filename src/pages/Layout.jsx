import { Outlet } from "react-router";
import Navbar from "../components/Navbar";
import MobileNavbar from "../components/MobileNavbar";
import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBars } from "@fortawesome/free-solid-svg-icons";

export default function HomeLayout() {
    const [navbarVisible, setNavbarVisible] = useState(false);
    return (
        <>
            <div className="hidden lg:block fixed top-0 left-0 w-screen h-fit z-50">
                <Navbar />
            </div>
            <div className={`${navbarVisible ? "block" : "hidden"} lg:hidden`}>
                <MobileNavbar onClose={() => { setNavbarVisible(false) }} />

            </div>
            {!navbarVisible && <div
                className="block lg:hidden fixed text-2xl top-0 left-0 text-white z-[999] m-4 hover:scale-110 duration-300 transition-all justify-center items-center align-middle cursor-pointer"
                onClick={() => { setNavbarVisible(true) }}>
                <FontAwesomeIcon icon={faBars} />
            </div>}
            <Outlet />
        </>
    );
}
