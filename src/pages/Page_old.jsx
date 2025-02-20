import * as motion from "motion/react-client";
import { Variants } from "motion/react";
import React from "react";

export default function HomePage() {
    return (
        <main>
            <section className="flex flex-col items-center justify-center py-10">
                <h1 className="text-3xl font-bold">Features</h1>
            </section>
            <ScrollTriggered />
        </main>
    );
}

function ScrollTriggered() {
    return (
        <div style={container}>
            {features.map(({ emoji, text, hueA, hueB }, i) => (
                <Card key={i} i={i} emoji={emoji} text={text} hueA={hueA} hueB={hueB} />
            ))}
        </div>
    );
}

interface CardProps {
    emoji: string;
    text: string;
    hueA: number;
    hueB: number;
    i: number;
}

function Card({ emoji, text, hueA, hueB, i }: CardProps) {
    const background = `linear-gradient(306deg, hsl(${hueA}, 100%, 50%), hsl(${hueB}, 100%, 50%))`;

    return (
        <motion.div
            className={`card-container-${i}`}
            style={cardContainer}
            initial="offscreen"
            whileInView="onscreen"
            viewport={{ amount: 0.8 }}
        >
            <div style={{ ...splash, background }} />
            <motion.div style={card} variants={cardVariants} className="card">
                <div className="text-center flex flex-col items-center">
                    <div className="text-6xl mb-2">{emoji}</div>
                    <p className="text-lg">{text}</p>
                </div>
            </motion.div>
        </motion.div>
    );
}

const cardVariants: Variants = {
    offscreen: {
        y: 300,
    },
    onscreen: {
        y: 50,
        rotate: -10,
        transition: {
            type: "spring",
            bounce: 0.4,
            duration: 0.8,
        },
    },
};

const container: React.CSSProperties = {
    margin: "100px auto",
    maxWidth: 500,
    paddingBottom: 100,
    width: "100%",
};

const cardContainer: React.CSSProperties = {
    overflow: "hidden",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    position: "relative",
    paddingTop: 20,
    marginBottom: -120,
};

const splash: React.CSSProperties = {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
};

const card: React.CSSProperties = {
    fontSize: "2rem",
    width: 300,
    height: 430,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    borderRadius: 20,
    background: "#f5f5f5",
    boxShadow:
        "0 0 1px hsl(0deg 0% 0% / 0.075), 0 0 2px hsl(0deg 0% 0% / 0.075), 0 0 4px hsl(0deg 0% 0% / 0.075), 0 0 8px hsl(0deg 0% 0% / 0.075), 0 0 16px hsl(0deg 0% 0% / 0.075)",
    textAlign: "center",
    padding: "20px",
};

const features = [
    { emoji: "📤", text: "Upload photos easily", hueA: 340, hueB: 10 },
    { emoji: "🔍", text: "Analyze images effectively", hueA: 20, hueB: 40 },
    { emoji: "📞", text: "Contact support anytime", hueA: 60, hueB: 90 },
    { emoji: "🔒", text: "Good Data Security", hueA: 100, hueB: 140 },
    { emoji: "⚙️", text: "Customize settings", hueA: 205, hueB: 245 },
    { emoji: "📂", text: "Manage files effortlessly", hueA: 260, hueB: 290 },
    { emoji: "📊", text: "View detailed analytics and support", hueA: 290, hueB: 320 },
];
