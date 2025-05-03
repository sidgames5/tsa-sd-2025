import React, { useState, useEffect, useRef } from "react";
import { useCookies } from "react-cookie";
import { Send, MessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router";

export default function AIChatbot() {
    const [cookies, setCookies] = useCookies(["darkMode", "user"]);
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const navigate = useNavigate();

    const isDark = cookies.darkMode;
    const user = cookies.user;


    if (!user) {
        return (
            <div className="flex justify-center items-center h-screen bg-gray-100 text-gray-800 dark:text-white">
                <p className="text-lg font-semibold">You need to log in to view this page.</p>
            </div>
        );
    }

    const handleSend = async () => {
        if (!query.trim()) return;

        const userMessage = { sender: "user", text: query };
        setMessages((prev) => [...prev, userMessage]);
        setQuery("");
        setLoading(true);

        try {
            const response = await fetch("api/ollama-support", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: query }),
            });
            const data = await response.json();
            const botMessage = { sender: "bot", text: data.reply };
            setMessages((prev) => [...prev, botMessage]);
        } catch (error) {
            console.error("Support error:", error);
            setMessages((prev) => [
                ...prev,
                { sender: "bot", text: "Sorry, something went wrong." },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const renderFormattedText = (text, keyPrefix) => {
        const parts = text.split(/(\*\*.*?\*\*)/g);
        return parts.map((part, i) => {
            if (part.startsWith("**") && part.endsWith("**")) {
                return <strong key={`${keyPrefix}-b-${i}`}>{part.slice(2, -2)}</strong>;
            }
            return <span key={`${keyPrefix}-s-${i}`}>{part}</span>;
        });
    };

    const formatBotText = (text) => {
        const lines = text.split("\n").filter(line => line.trim() !== "");
        const formatted = [];
        let currentList = [];

        lines.forEach((line, index) => {
            const trimmed = line.trim();

            if (trimmed.startsWith("-")) {
                currentList.push(trimmed.replace(/^-\s*/, ""));
            } else {
                if (currentList.length > 0) {
                    formatted.push(
                        <ul className="list-disc list-inside space-y-1 mb-2" key={`ul-${index}`}>
                            {currentList.map((item, idx) => (
                                <li key={`li-${idx}`}>{renderFormattedText(item, `li-${idx}`)}</li>
                            ))}
                        </ul>
                    );
                    currentList = [];
                }

                if (trimmed.endsWith(":")) {
                    formatted.push(
                        <p key={`bold-${index}`} className="font-semibold mb-1">
                            {renderFormattedText(trimmed.slice(0, -1), `h-${index}`)}
                        </p>
                    );
                } else {
                    formatted.push(
                        <p key={`p-${index}`} className="mb-1">
                            {renderFormattedText(trimmed, `p-${index}`)}
                        </p>
                    );
                }
            }
        });

        if (currentList.length > 0) {
            formatted.push(
                <ul className="list-disc list-inside space-y-1 mb-2" key={`ul-final`}>
                    {currentList.map((item, idx) => (
                        <li key={`li-final-${idx}`}>{renderFormattedText(item, `li-final-${idx}`)}</li>
                    ))}
                </ul>
            );
        }

        return formatted;
    };

    return (
        <div className="fixed bottom-6 right-6 z-50">
            <motion.button
                onClick={() => setShowModal(!showModal)}
                className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-xl"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
            >
                <MessageSquare size={24} />
            </motion.button>
            <AnimatePresence>
                {showModal && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8, y: 50 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.8, y: 50 }}
                        transition={{ dration: 0.3 }}
                        className={`absolute flex flex-col bottom-16 right-0 w-96 max-h-[60vh] shadow-2xl rounded-2xl p-4 gap-4 overflow-hidden border-2 ${cookies.darkMode ? "bg-gradient-to-br from-indigo-950 via-sky-700 to-indigo-950 border-white" : "border-black bg-white"}`}
                        drag
                    >
                        <h1 className={`${cookies.darkMode ? "text-white" : "text-black"} mt-3 text-2xl font-bold text-center`}>Support Assistant</h1>

                        <div className={`h-[1px] w-full ${cookies.darkMode ? "bg-white" : "bg-black"}`}></div>

                        <div className="flex-grow overflow-y-auto">
                            {messages.map((msg, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.3 }}
                                    className={`rounded-xl px-4 py-2 w-fit max-w-[85%] whitespace-pre-wrap mt-2 ${msg.sender === "user" ? "ml-auto bg-blue-500 text-black" : "bg-gray-200"} text-black`}
                                >
                                    {msg.sender === "bot" ? formatBotText(msg.text) : msg.text}
                                </motion.div>
                            ))}
                            {loading && <div className="text-sm text-center text-gray-500">Typing...</div>}
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="text"
                                placeholder="Ask a question..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                className={`flex-grow px-4 py-2 rounded-xl border ${cookies.darkMode ? "bg-black text-white border-white" : "bg-white text-black border-black"}`}
                                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                            />
                            <button
                                onClick={handleSend}
                                className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl"
                            >
                                <Send size={20} />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}