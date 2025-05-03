import React, { useState, useEffect } from "react";
import { useCookies } from "react-cookie";
import { Send } from "lucide-react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router";

export default function SupportPage() {
    const [cookies, setCookies] = useCookies(["darkMode", "user"]);
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const isDark = cookies.darkMode;
    const user = cookies.user;

    useEffect(() => {
        if (!user) {
            // If the user is not logged in, redirect to login page
            navigate("/login");
        }
    }, [user, navigate]);

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
        <main className={`mt-20 ${isDark ? "bg-gray-900 text-gray-100" : "bg-gray-50 text-gray-900"} w-full min-h-screen py-20 px-6`}>
            <div className="max-w-3xl mx-auto">
                <h1 className="text-4xl font-bold text-center mb-10">Support Assistant</h1>

                {user ? (
                    <>
                        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 space-y-4 max-h-[70vh] overflow-y-auto mb-6">
                            {messages.map((msg, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ duration: 0.3 }}
                                    className={`rounded-xl px-4 py-2 w-fit max-w-[80%] whitespace-pre-wrap ${msg.sender === "user" ? "ml-auto bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
                                >
                                    {msg.sender === "bot" ? formatBotText(msg.text) : msg.text}
                                </motion.div>
                            ))}
                            {loading && <div className="text-sm text-gray-500 italic">Thinking...</div>}
                        </div>

                        <div className="flex gap-4 items-center">
                            <input
                                type="text"
                                placeholder="Ask a farming question..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                className={`${cookies.darkMode ? "text-white" : "text-gray-950"} flex-grow px-4 py-2 rounded-xl border border-gray-300 dark:border-gray-600 dark:bg-gray-800`}
                                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                            />
                            <button
                                onClick={handleSend}
                                className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition"
                            >
                                <Send size={20} />
                            </button>
                        </div>
                    </>
                ) : (
                    <div className="text-center">
                        <p>You must be logged in to use the support chat.</p>
                    </div>
                )}
            </div>
        </main>
    );
}
