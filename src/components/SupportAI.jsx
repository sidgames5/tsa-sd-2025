import React, { useState, useEffect, useRef } from "react";
import { useCookies } from "react-cookie";
import { Send, MessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {faTimes, faSpinner} from "@fortawesome/free-solid-svg-icons"
export default function AIChatbot() {
    const [cookies, setCookies] = useCookies(["darkMode", "user"]);
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const navigate = useNavigate();

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
            {!user ? (
                <motion.button
                    onClick={() => navigate('/login')}
                    className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-xl"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                    title="Login to use AI Assistant"
                >
                    <MessageSquare size={24} />
                </motion.button>
            ) : (
                <>
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
                                transition={{ duration: 0.3 }}
                                className={`absolute flex flex-col bottom-16 right-0 w-96 max-h-[60vh] shadow-2xl rounded-2xl p-4 gap-4 overflow-hidden border-2 ${
                                    isDark 
                                        ? "bg-gradient-to-br from-indigo-950 via-sky-800 to-indigo-950 border-white" 
                                        : "border-black bg-white"
                                }`}
                            >
                                <div className="flex justify-between items-center">
                                    <h1 className={`${isDark ? "text-white" : "text-black"} text-xl font-bold`}>
                                        LeafLogic AI
                                    </h1>
                                    <button 
                                        onClick={() => setShowModal(false)}
                                        className={`p-1 rounded-full ${isDark ? "hover:bg-gray-700" : "hover:bg-gray-200"}`}
                                    >
                                        <FontAwesomeIcon icon={faTimes} />
                                    </button>
                                </div>

                                <div className={`h-[1px] w-full ${isDark ? "bg-white" : "bg-black"}`}></div>

                                <div className="flex-grow overflow-y-auto">
                                    {messages.map((msg, idx) => (
                                        <motion.div
                                            key={idx}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.3 }}
                                            className={`rounded-xl px-4 py-2 w-fit max-w-[85%] whitespace-pre-wrap mt-2 ${
                                                msg.sender === "user" 
                                                    ? "ml-auto bg-blue-500 text-white" 
                                                    : isDark 
                                                        ? "bg-gray-700 text-white" 
                                                        : "bg-gray-200 text-black"
                                            }`}
                                        >
                                            {msg.sender === "bot" ? formatBotText(msg.text) : msg.text}
                                        </motion.div>
                                    ))}
                                    {loading && (
                                        <div className={`${isDark ? "text-gray-300" : "text-gray-500"} text-sm text-center py-2`}>
                                            <FontAwesomeIcon icon={faSpinner} className="animate-spin mr-2" />
                                            Thinking...
                                        </div>
                                    )}
                                </div>

                                <div className="flex items-center gap-2">
                                    <input
                                        type="text"
                                        placeholder="Ask a question..."
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        className={`flex-grow px-4 py-2 rounded-xl border ${
                                            isDark 
                                                ? "bg-gray-700 text-white border-gray-600" 
                                                : "bg-white text-black border-gray-300"
                                        }`}
                                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                                    />
                                    <button
                                        onClick={handleSend}
                                        disabled={loading}
                                        className={`p-2 rounded-xl ${
                                            loading
                                                ? "bg-gray-500 cursor-not-allowed"
                                                : "bg-blue-600 hover:bg-blue-700"
                                        } text-white`}
                                    >
                                        <Send size={20} />
                                    </button>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </>
            )}
        </div>
    );
}