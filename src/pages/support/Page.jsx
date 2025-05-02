import React, { useState } from "react";
import { useCookies } from "react-cookie";
import { Send, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function SupportPageModal({ isOpen, onClose }) {
    const [cookies] = useCookies(["darkMode", "user"]);
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const isDark = cookies.darkMode;
    const user = cookies.user;

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
                                <li key={idx}>{renderFormattedText(item, `li-${idx}`)}</li>
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
                        <li key={idx}>{renderFormattedText(item, `li-final-${idx}`)}</li>
                    ))}
                </ul>
            );
        }

        return formatted;
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                >
                    <motion.div
                        className={`w-full max-w-3xl max-h-[90vh] overflow-y-auto p-6 rounded-2xl shadow-2xl ${isDark ? "bg-gray-900 text-white" : "bg-white text-black"}`}
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                    >
                        <div className="flex justify-between items-center mb-4">
                            <h1 className="text-3xl font-bold">Support Assistant</h1>
                            <button onClick={onClose} className="text-gray-500 hover:text-red-500">
                                <X size={24} />
                            </button>
                        </div>

                        {!user ? (
                            <p>You need to log in to use the support chat.</p>
                        ) : (
                            <>
                                <div className="rounded-xl p-4 border dark:border-gray-700 max-h-[50vh] overflow-y-auto space-y-2 mb-4 bg-gray-100 dark:bg-gray-800">
                                    {messages.map((msg, idx) => (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.3 }}
                                            className={`rounded-xl px-4 py-2 w-fit max-w-[80%] whitespace-pre-wrap ${msg.sender === "user" ? "ml-auto bg-blue-500 text-white" : "bg-gray-300 dark:bg-gray-700"}`}
                                        >
                                            {msg.sender === "bot" ? formatBotText(msg.text) : msg.text}
                                        </motion.div>
                                    ))}
                                    {loading && (
                                        <motion.div
                                            className="w-full h-1 bg-blue-200 dark:bg-gray-600 rounded overflow-hidden mb-2"
                                            initial={{ width: 0 }}
                                            animate={{ width: "100%" }}
                                            transition={{ duration: 2, ease: "easeInOut", repeat: Infinity }}
                                        >
                                            <motion.div
                                                className="h-full bg-blue-600"
                                                initial={{ x: "-100%" }}
                                                animate={{ x: "100%" }}
                                                transition={{
                                                    duration: 1.5,
                                                    ease: "linear",
                                                    repeat: Infinity,
                                                }}
                                            />
                                        </motion.div>
                                    )}
                                </div>

                                <div className="flex gap-4">
                                    <input
                                        type="text"
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                                        placeholder="Ask something..."
                                        className="flex-grow px-4 py-2 rounded-xl border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
                                    />
                                    <button
                                        onClick={handleSend}
                                        className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl"
                                    >
                                        <Send size={20} />
                                    </button>
                                </div>
                            </>
                        )}
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
