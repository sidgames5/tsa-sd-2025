import * as motion from "motion/react-client";
import { Variants } from "motion/react";
import React from "react";

function HoverPopupCard({ children, title, color }) {
    return <div className={`flex flex-col bg-${color} w-96 h-[32rem] p-4 rounded-3xl relative items-center`}>
        <motion.div className="flex flex-col items-center bg-white w-72 h-[21rem] p-4 rounded-3xl absolute bottom-4"
            initial={{ translateY: 0, rotate: -1 }}
            whileHover={{ translateY: -125, rotate: -3 }}
            transition={{ type: "spring", stiffness: 150 }}>
            <span className="text-8xl">{title}</span>
            <div className="text-xl mt-8">{children}</div>
        </motion.div>
        <div className={`absolute bg-${color} bottom-0 w-96 h-[11rem] rounded-3xl pointer-events-none`}>
            <div className={`absolute bg-gradient-to-t from-${color} to-transparent h-[4rem] w-96 bottom-full`}>&nbsp;</div>
        </div>
    </div>;
}

export default function HomePage() {
    const cardItems = [
        { emoji: "📤", text: "Upload photos easily", color: "slate-600" },
        { emoji: "🔍", text: "Analyze images effectively", color: "slate-600" },
        { emoji: "📞", text: "Contact support anytime", color: "slate-600" },
        { emoji: "🔒", text: "Good Data Security", color: "slate-600" },
        { emoji: "⚙️", text: "Customize settings", color: "slate-600" },
        { emoji: "📂", text: "Manage files effortlessly", color: "slate-600" },
        { emoji: "📊", text: "View detailed analytics and support", color: "slate-600" },
    ];

    return <main>
        <div className="flex flex-col items-center justify-center py-10 gap-10">
            <h1 className="text-3xl font-bold">Features</h1>

            <div className="grid grid-cols-3 gap-16">
                {cardItems.map((item) => <HoverPopupCard title={item.emoji} color={item.color}>{item.text}</HoverPopupCard>)}
            </div>
        </div>
    </main>;
}