import React from "react";
import { useCookies } from "react-cookie";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";

export function FeatureCard({ icon, description, delay = 0 }) {
    const [cookies] = useCookies(["darkMode"]);
    const isDark = cookies.darkMode;
    const bg = isDark ? "bg-gray-800/60 backdrop-blur-md" : "bg-gradient-to-br from-green-50 via-white to-emerald-50/70 backdrop-blur-lg";
    const text = isDark ? "text-white" : "text-gray-900";
    const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.1 });

    return (
        <motion.div
            ref={ref}
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, ease: "easeOut" }}
            whileHover={{
                scale: 1.04,
                rotate: 0.5,
                boxShadow: isDark
                    ? "0 10px 30px rgba(255, 255, 255, 0.1)"
                    : "0 10px 30px rgba(0, 0, 0, 0.1)",
                transition: { duration: 0.3 }
            }}
            className={`transition-all duration-300 ${bg} ${text} rounded-3xl p-6 min-h-[220px] flex flex-col items-start gap-4 border border-white/20 shadow-xl`}
        >
            <div className="text-4xl">{icon}</div>
            <p className="text-md sm:text-lg leading-relaxed">{description}</p>
        </motion.div>
    );
}

export default function FeaturesPage() {
    const [cookies] = useCookies(["darkMode"]);

    const cardItems = [
        { emoji: "📤", text: "Upload plant images quickly and easily." },
        { emoji: "🔍", text: "AI-powered image analysis for accurate diagnosis." },
        { emoji: "📞", text: "Dedicated support team available to assist anytime." },
        { emoji: "🔒", text: "Secure data handling with robust privacy practices." },
        { emoji: "⚙️", text: "Customizable settings to tailor the experience." },
        { emoji: "📂", text: "Organize and manage your plant image library." },
        { emoji: "📊", text: "Access comprehensive visual reports and analytics." },
        { emoji: "🧭", text: "Simple, intuitive navigation throughout the app." },
        { emoji: "👵", text: "User-friendly interface for all experience levels." }
    ];

    return (
        <main className={`${cookies.darkMode ? "bg-gray-900" : "bg-gray-100"} w-full h-[110vh]`}>
            <div className="max-w-6xl mx-auto px-4 mt-20"> 
                <h1 className={`text-4xl font-semibold p-12 text-center ${cookies.darkMode ? "text-white" : "text-green-900"}`}>
                    Key Features
                </h1>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                    {cardItems.map((item, index) => (
                        <FeatureCard
                            key={index}
                            icon={item.emoji}
                            description={item.text}
                            delay={index * 0.1} // stagger effect
                        />
                    ))}
                </div>
            </div>
        </main>
    );
}

