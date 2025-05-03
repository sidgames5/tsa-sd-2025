import React from "react";
import { useCookies } from "react-cookie";
import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import AIChatbot from "/Users/kaniskprakash/Documents/GitHub/tsa-sd-2025/src/components/SupportAI.jsx";


export function FeatureCard({ icon, description, delay = 0 }) {
    const [cookies] = useCookies(["darkMode"]);
    const isDark = cookies.darkMode;

    const bg = isDark
        ? "bg-gray-800/60 backdrop-blur-md"
        : "bg-gradient-to-br from-green-50 via-stone-50 to-emerald-50/80 backdrop-blur-md";
    const text = isDark ? "text-white" : "text-gray-800";

    const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.1 });

    return (
        <motion.div
            ref={ref}
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, ease: "easeOut", delay }}
            whileHover={{
                scale: 1.03,
                rotate: 0.3,
                boxShadow: isDark
                    ? "0 12px 24px rgba(255, 255, 255, 0.08)"
                    : "0 12px 24px rgba(0, 0, 0, 0.08)",
                transition: { duration: 0.3 }
            }}
            className={`transition-all duration-300 ${bg} ${text} rounded-2xl p-6 min-h-[200px] flex flex-col justify-start items-start gap-4 border border-white/10 shadow-md`}
        >
            <div className="text-5xl">{icon}</div>
            <p className="text-base sm:text-lg font-medium leading-relaxed">{description}</p>
        </motion.div>
    );
}

export default function FeaturesPage() {
    const [cookies] = useCookies(["darkMode"]);
    const isDark = cookies.darkMode;

    const cardItems = [
        { emoji: "📤", text: "Upload plant images quickly and easily." },
        { emoji: "🔍", text: "AI-powered image analysis for accurate diagnosis." },
        { emoji: "📞", text: "24/7 support from our expert team." },
        { emoji: "🔒", text: "Secure data handling with industry-grade encryption." },
        { emoji: "⚙️", text: "Flexible settings to personalize your workflow." },
        { emoji: "📂", text: "Effortlessly manage your plant image collections." },
        { emoji: "📊", text: "Detailed visual reports and insightful analytics." },
        { emoji: "🧭", text: "Clean, intuitive app navigation." },
        { emoji: "👵", text: "Accessible design for users of all experience levels." }
    ];

    return (
        <main className={`${isDark ? "bg-gray-900" : "bg-gray-50"} w-full min-h-screen py-16`}>
            <AIChatbot />
            <div className="max-w-6xl mx-auto px-4 mt-10">
                <h1
                    className={`text-4xl sm:text-5xl font-bold text-center mb-16 
                        bg-clip-text text-transparent p-8
                        ${isDark 
                            ? "bg-gradient-to-r from-blue-400 via-sky-500 to-indigo-500" 
                        : "bg-gradient-to-r from-green-500 via-emerald-600 to-teal-500"}`}
                    >
                    Key Features
                </h1>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                    {cardItems.map((item, index) => (
                        <FeatureCard
                            key={index}
                            icon={item.emoji}
                            description={item.text}
                            delay={index * 0.1}
                        />
                    ))}
                </div>
            </div>
        </main>
    );
}
