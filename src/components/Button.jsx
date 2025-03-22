import { motion } from "motion/react";

export default function Button({ children, onClick, className, disabled }) {
    return <motion.button
        initial={{ scale: 1 }}
        whileHover={{ scale: 1.05 }}
        transition={{ duration: 0.3 }}
        className={`bg-sky-600 p-4 rounded-lg text-white ${className}`}
        disabled={disabled}
        onClick={onClick}
    >
        {children}
    </motion.button>;
}

