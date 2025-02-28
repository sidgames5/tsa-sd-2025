import { motion } from "motion/react";

export default function NavbarButton({ children }) {
    return <div
        className="list-none text-2xl font-bold text-white border-2 border-white rounded-lg px-4 py-2 transition duration-300 hover:bg-white hover:text-gray-900 hover:shadow-blue-500/50 hover:translate-x-2"
    >
        {children}
    </div>;
}