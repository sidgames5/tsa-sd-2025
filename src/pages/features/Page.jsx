import * as motion from "motion/react-client";
import React from "react";

function HoverPopupCard({ children, title, color }) {
    console.log(`bg-${color}`);
    return <div className={`flex flex-col bg-gray-700 w-96 h-[28rem] p-4 rounded-3xl relative items-center`}>
        <motion.div className="flex flex-col items-center bg-white w-72 h-[17rem] p-4 rounded-3xl absolute bottom-4"
            initial={{ translateY: 0, rotate: -1 }}
            whileHover={{ translateY: -125, rotate: -4 }}
            transition={{ type: "spring", stiffness: 90, damping: 15 }}>
            <span className="text-8xl">{title}</span>
            <div className="text-xl mt-8 text-center">{children}</div>
        </motion.div>
        <div className={`absolute bg-gray-700 bottom-0 w-96 h-[7rem] rounded-3xl pointer-events-none`}>
            <div className={`absolute bg-gradient-to-t from-gray-700 to-transparent h-[4rem] w-96 bottom-full`}>&nbsp;</div>
        </div>
    </div>;
}

export default function FeaturesPage() {
    const cardItems = [
        { emoji: "📤", text: "Upload photos easily", color: "slate-600" },
        { emoji: "🔍", text: "Analyze images effectively", color: "slate-600" },
        { emoji: "📞", text: "Contact support anytime", color: "slate-600" },
        { emoji: "🔒", text: "Good Data Security", color: "slate-600" },
        { emoji: "⚙️", text: "Customize settings", color: "slate-600" },
        { emoji: "📂", text: "Manage files effortlessly", color: "slate-600" },
        { emoji: "📊", text: "View detailed analytics and support", color: "slate-600" },
        { emoji: "🧭", text: "Easy to navigate", color: "slate-600" },
        { emoji: "👵", text: "User friendly", color: "slat-600"}
    ];

    return <main>
        <div className="flex flex-col items-center justify-center mt-[8vh] py-10 gap-10">
            <h1 className="text-5xl font-bold text-gray-200">Features</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-8">
                {cardItems.map((item) => <HoverPopupCard title={item.emoji} color={item.color}>{item.text}</HoverPopupCard>)}
            </div>
        </div>
    </main>;
}