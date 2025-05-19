import React, { useRef, useState } from "react";
import { useCookies } from "react-cookie";
import { motion, useScroll, useTransform, AnimatePresence } from "framer-motion";
import { useInView } from "react-intersection-observer";
import AIChatbot from "../../components/AIChatbot";


export default function FeaturesPage() {
    const [cookies] = useCookies(["darkMode"]);
    const isDark = cookies.darkMode;
    const containerRef = useRef(null);
    const { scrollYProgress } = useScroll({ target: containerRef });
    const y = useTransform(scrollYProgress, [0, 1], [0, -100]);
    const [showDemoModal, setShowDemoModal] = useState(false);
    const [currentDemoStep, setCurrentDemoStep] = useState(0);

    const features = [
        {
            emoji: "🌱",
            title: "Plant Upload",
            text: "Drag & drop or capture photos of your plants",
            color: isDark ? "from-emerald-900/50 to-emerald-700/50" : "from-emerald-100 to-emerald-200"
        },
        {
            emoji: "🔍",
            title: "AI Analysis",
            text: "Instant disease detection with very high accuracy",
            color: isDark ? "from-blue-900/50 to-blue-700/50" : "from-blue-100 to-blue-200"
        },
        {
            emoji: "📊",
            title: "Health Reports",
            text: "Detailed analytics and treatment plans",
            color: isDark ? "from-purple-900/50 to-purple-700/50" : "from-purple-100 to-purple-200"
        },
        {
            emoji: "🔄",
            title: "Progress Tracking",
            text: "Monitor recovery with timeline visuals",
            color: isDark ? "from-amber-900/50 to-amber-700/50" : "from-amber-100 to-amber-200"
        },
        {
            emoji: "📱",
            title: "Mobile Ready",
            text: "Full functionality on all devices",
            color: isDark ? "from-cyan-900/50 to-cyan-700/50" : "from-cyan-100 to-cyan-200"
        },
        {
            emoji: "🔒",
            title: "Secure Storage",
            text: "Encrypted cloud backup for your data",
            color: isDark ? "from-violet-900/50 to-violet-700/50" : "from-violet-100 to-violet-200"
        }
    ];
    const demoSteps = [
        {
            title: "Upload Process",
            description: "Drag and drop plant images or use our mobile camera capture",
            animation: "📤",
            tips: [
                "Use clear, well-lit photos of leaves",
                "Capture both healthy and affected areas",
                "Specify the plant type of the image you are uploading"
            ]
        },
        {
            title: "AI Analysis",
            description: "Our AI Model processes images in real-time",
            animation: "🤖",
            tips: [
                "Very high accuracy for common plant types",
                "Processes in under 10 seconds",
                "Very reliable"
            ]
        },
        {
            title: "Results Dashboard",
            description: "Interactive visualization of plant health metrics and the model's confidence vs accuracy on all your individual scans",
            animation: "📊",
            tips: [
                "Graph of your confidence vs how wrong the model thinks it is",
                "Historical trends over time",
                "Ability to email yourself your results"
            ]
        },
        {
            title: "Treatment For Your Plans",
            description: "Tips to combat any health problem for your plants",
            animation: "💊",
            tips: [
                "Organic and chemical options",
                "Step-by-step instructions",
                "Preventative care suggestions"
            ]
        }
    ];
    const FeatureCard = ({ icon, title, description, color, index }) => {
        const [ref, inView] = useInView({ threshold: 0.2, triggerOnce: true });

        return (
            <motion.div
                ref={ref}
                initial={{ opacity: 0, y: 50, rotateX: -45 }}
                animate={inView ? {
                    opacity: 1,
                    y: 0,
                    rotateX: 0,
                    transition: {
                        type: "spring",
                        stiffness: 100,
                        damping: 15,
                        delay: index * 0.15
                    }
                } : {}}
                whileHover={{
                    y: -10,
                    rotateY: 5,
                    scale: 1.03,
                    transition: { duration: 0.3 }
                }}
                style={{
                    transformStyle: "preserve-3d",
                    perspective: "1000px"
                }}
                className={`rounded-2xl p-8 h-full flex flex-col bg-gradient-to-br ${color} shadow-lg hover:shadow-xl border ${isDark ? "border-gray-700" : "border-gray-200"
                    }`}
            >
                <motion.div
                    whileHover={{ scale: 1.2, rotate: 5 }}
                    className="text-5xl mb-6 text-center"
                    style={{ transformStyle: "preserve-3d" }}
                >
                    {icon}
                </motion.div>
                <h3 className={`text-2xl font-bold mb-3 text-center ${isDark ? "text-white" : "text-gray-800"
                    }`}>{title}</h3>
                <p className={`text-lg ${isDark ? "text-gray-300" : "text-gray-600"
                    } text-center`}>{description}</p>
            </motion.div>
        );
    };
    const handleDemoClick = () => {
        setCurrentDemoStep(0);
        setShowDemoModal(true);
    }

    const handleNextStep = () => {
        setCurrentDemoStep((prev) => (prev + 1) % demoSteps.length);
    }

    const handlePrevStep = () => {
        setCurrentDemoStep((prev) => (prev - 1 + demoSteps.length) % demoSteps.length);
    }

    return (
        <div
            ref={containerRef}
            className={`min-h-screen w-full overflow-hidden mt-20 ${isDark ? "bg-gray-950" : "bg-gradient-to-b from-gray-50 to-gray-100"
                }`}
        >
            {/* Floating background elements */}
            <motion.div
                style={{ y }}
                className="fixed inset-0 overflow-hidden pointer-events-none"
            >
                <div className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full bg-emerald-500/10 blur-3xl"></div>
                <div className="absolute top-2/3 right-1/3 w-96 h-96 rounded-full bg-blue-500/10 blur-3xl"></div>
                <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-purple-500/10 blur-3xl"></div>
            </motion.div>

            <div className="relative z-10 container mx-auto px-4 py-24 max-w-7xl">
                {/* Animated header section */}
                <motion.div
                    initial={{ opacity: 0, y: -50 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                    className="text-center mb-24"
                >
                    <motion.h1
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3, duration: 0.8 }}
                        className={`text-5xl md:text-6xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"
                            }`}
                    >
                        Transform Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-500">Plant Care</span>
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.6, duration: 0.8 }}
                        className={`text-xl max-w-3xl mx-auto ${isDark ? "text-gray-400" : "text-gray-600"
                            }`}
                    >
                        Advanced tools powered by AI to keep your plants thriving
                    </motion.p>
                </motion.div>

                {/* 3D Feature Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-32">
                    {features.map((feature, index) => (
                        <FeatureCard
                            key={index}
                            index={index}
                            icon={feature.emoji}
                            title={feature.title}
                            description={feature.text}
                            color={feature.color}
                        />
                    ))}
                </div>

                {/* Enhanced Interactive Demo Section */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8 }}
                    viewport={{ once: true }}
                    className={`rounded-3xl p-8 md:p-16 mb-24 ${isDark ? "bg-gray-900/50 backdrop-blur-md" : "bg-white/80 backdrop-blur-sm"
                        } border ${isDark ? "border-gray-800" : "border-gray-200"
                        } shadow-2xl`}
                >
                    <div className="flex flex-col lg:flex-row items-center gap-12">
                        <div className="flex-1">
                            <h2 className={`text-3xl md:text-4xl font-bold mb-6 ${isDark ? "text-white" : "text-gray-900"
                                }`}>
                                See It In <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-500">Action</span>
                            </h2>
                            <p className={`text-lg mb-8 ${isDark ? "text-gray-300" : "text-gray-600"
                                }`}>
                                Our interactive demo shows how quickly you can diagnose plant health issues with our AI technology.
                            </p>

                            {/* Enhanced Live Demo Button */}
                            <motion.button
                                whileHover={{
                                    scale: 1.05,
                                    boxShadow: isDark
                                        ? "0 0 20px rgba(16, 185, 129, 0.5)"
                                        : "0 0 20px rgba(5, 150, 105, 0.3)"
                                }}
                                whileTap={{ scale: 0.95 }}
                                onClick={handleDemoClick}
                                className={`px-8 py-4 rounded-full font-bold text-lg ${isDark
                                    ? "bg-gradient-to-r from-emerald-500 to-teal-600 text-white"
                                    : "bg-gradient-to-r from-emerald-400 to-teal-500 text-white"
                                    } shadow-lg hover:shadow-xl transition-all relative overflow-hidden group`}
                            >
                                <span className="relative z-10">Try Live Demo</span>
                                <motion.span
                                    initial={{ x: -100, opacity: 0 }}
                                    animate={{
                                        x: ["-100%", "100%"],
                                        opacity: [0, 0.3, 0]
                                    }}
                                    transition={{
                                        duration: 2.5,
                                        repeat: Infinity,
                                        ease: "easeInOut"
                                    }}
                                    className="absolute inset-0 bg-white/30 z-0"
                                    style={{
                                        transform: "skewX(-20deg)"
                                    }}
                                />
                            </motion.button>
                        </div>

                        <div className="flex-1">
                            <motion.div
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                style={{
                                    transformStyle: "preserve-3d",
                                    transform: "rotateX(5deg) rotateY(-5deg)"
                                }}
                                className={`rounded-2xl overflow-hidden border-2 ${isDark ? "border-gray-700" : "border-gray-200"
                                    } shadow-lg`}
                            >
                                <div className={`aspect-video bg-gradient-to-br ${isDark ? "from-gray-800 to-gray-900" : "from-gray-100 to-gray-200"
                                    } flex items-center justify-center`}>
                                    <img src="https://cdn.pixabay.com/photo/2019/07/14/16/27/pen-4337521_1280.jpg" alt="Computer Image" />
                                </div>
                            </motion.div>
                        </div>
                    </div>
                </motion.div>

                {/* 3D Demo Modal */}
                <AnimatePresence>
                    {showDemoModal && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="fixed inset-0 z-50 flex items-center justify-center p-4"
                        >
                            {/* Backdrop */}
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 0.7 }}
                                exit={{ opacity: 0 }}
                                onClick={() => setShowDemoModal(false)}
                                className="absolute inset-0 bg-black"
                            />

                            {/* Modal Content */}
                            <motion.div
                                initial={{
                                    opacity: 0,
                                    scale: 0.8,
                                    rotateY: 90,
                                    perspective: "1000px"
                                }}
                                animate={{
                                    opacity: 1,
                                    scale: 1,
                                    rotateY: 0,
                                    transition: {
                                        type: "spring",
                                        stiffness: 100,
                                        damping: 20
                                    }
                                }}
                                exit={{
                                    opacity: 0,
                                    scale: 0.8,
                                    rotateY: -90,
                                    transition: { duration: 0.3 }
                                }}
                                className={`relative z-10 rounded-3xl p-8 max-w-4xl w-full ${isDark ? "bg-gray-900" : "bg-white"
                                    } shadow-2xl border ${isDark ? "border-gray-700" : "border-gray-200"
                                    }`}
                                style={{
                                    transformStyle: "preserve-3d"
                                }}
                            >
                                {/* Close Button */}
                                <button
                                    onClick={() => setShowDemoModal(false)}
                                    className={`absolute top-4 right-4 p-2 rounded-full ${isDark ? "hover:bg-gray-800" : "hover:bg-gray-100"
                                        }`}
                                >
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>

                                {/* Demo Content */}
                                <div className="flex flex-col lg:flex-row gap-8">
                                    {/* 3D Animation Panel */}
                                    <motion.div
                                        key={currentDemoStep}
                                        initial={{ rotateX: -15, rotateY: -15 }}
                                        animate={{
                                            rotateX: 0,
                                            rotateY: 0,
                                            transition: {
                                                type: "spring",
                                                stiffness: 50,
                                                damping: 10
                                            }
                                        }}
                                        className="flex-1 flex items-center justify-center"
                                        style={{
                                            transformStyle: "preserve-3d",
                                            perspective: "1000px"
                                        }}
                                    >
                                        <div className={`text-9xl p-12 rounded-2xl ${isDark ? "bg-gray-800" : "bg-gray-100"
                                            } shadow-lg`}>
                                            {demoSteps[currentDemoStep].animation}
                                        </div>
                                    </motion.div>

                                    <div className={`w-[1px] ${isDark ? "bg-white" : "bg-black"} `}></div>
                                    {/* Info Panel */}
                                    <div className="flex-1">
                                        <motion.h3
                                            key={`title-${currentDemoStep}`}
                                            initial={{ x: 20, opacity: 0 }}
                                            animate={{ x: 0, opacity: 1 }}
                                            className={`text-3xl font-bold mb-4 ${isDark ? "text-emerald-400" : "text-emerald-600"
                                                }`}
                                        >
                                            {demoSteps[currentDemoStep].title}
                                        </motion.h3>

                                        <motion.p
                                            key={`desc-${currentDemoStep}`}
                                            initial={{ x: 20, opacity: 0 }}
                                            animate={{ x: 0, opacity: 1 }}
                                            transition={{ delay: 0.1 }}
                                            className={`text-lg mb-6 ${isDark ? "text-gray-300" : "text-gray-600"
                                                }`}
                                        >
                                            {demoSteps[currentDemoStep].description}
                                        </motion.p>

                                        <div className="mb-8">
                                            <h4 className={`text-sm font-semibold mb-3 ${isDark ? "text-gray-400" : "text-gray-500"
                                                } uppercase tracking-wider`}>
                                                Pro Tips
                                            </h4>
                                            <ul className="space-y-2">
                                                {demoSteps[currentDemoStep].tips.map((tip, i) => (
                                                    <motion.li
                                                        key={i}
                                                        initial={{ x: 10, opacity: 0 }}
                                                        animate={{
                                                            x: 0,
                                                            opacity: 1,
                                                            transition: { delay: 0.2 + i * 0.05 }
                                                        }}
                                                        className={`flex items-start ${isDark ? "text-gray-300" : "text-gray-700"
                                                            }`}
                                                    >
                                                        <span className={`inline-block mr-2 mt-1 ${isDark ? "text-emerald-400" : "text-emerald-500"
                                                            }`}>•</span>
                                                        {tip}
                                                    </motion.li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Navigation Controls */}
                                        <div className="flex justify-between mt-6">
                                            <motion.button
                                                whileHover={{ x: -2 }}
                                                whileTap={{ x: -5 }}
                                                onClick={handlePrevStep}
                                                className={`px-4 py-2 rounded-lg ${isDark ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-800 hover:bg-gray-900"
                                                    }`}
                                            >
                                                Previous
                                            </motion.button>
                                            <div className="flex space-x-1">
                                                {demoSteps.map((_, i) => (
                                                    <button
                                                        key={i}
                                                        onClick={() => setCurrentDemoStep(i)}
                                                        className={`w-3 h-3 rounded-full ${currentDemoStep === i
                                                            ? isDark ? "bg-emerald-400" : "bg-emerald-500"
                                                            : isDark ? "bg-gray-700" : "bg-gray-300"
                                                            }`}
                                                    />
                                                ))}
                                            </div>
                                            <motion.button
                                                whileHover={{ x: 2 }}
                                                whileTap={{ x: 5 }}
                                                onClick={handleNextStep}
                                                className={`px-4 py-2 rounded-lg ${isDark ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-800 hover:bg-gray-900"
                                                    }`}
                                            >
                                                Next
                                            </motion.button>
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            <AIChatbot />
        </div>
    );
}
