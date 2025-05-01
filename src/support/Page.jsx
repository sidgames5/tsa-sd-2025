import React, { useState } from "react";
import { useCookies } from "react-cookie";
import { Send } from "lucide-react";
import { motion } from "framer-motion";

export default function SupportPage() {
    const [cookies] = useCookies(["darkMode"]);
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const isDark = cookies.darkMode;

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

    return (
        <main className={`mt-20 ${isDark ? "bg-gray-900 text-gray-100" : "bg-gray-50 text-gray-900"} w-full min-h-screen py-20 px-6`}>
            <div className="max-w-3xl mx-auto">
                <h1 className="text-4xl font-bold text-center mb-10">Support Assistant</h1>
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 space-y-4 max-h-[70vh] overflow-y-auto mb-6">
                    {messages.map((msg, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className={`rounded-xl px-4 py-2 w-fit max-w-[80%] ${msg.sender === "user" ? "ml-auto bg-blue-500 text-white" : "bg-gray-200 dark:bg-gray-700"}`}
                        >
                            {msg.text}
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
                        className="flex-grow px-4 py-2 rounded-xl border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    />
                    <button
                        onClick={handleSend}
                        className="p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition"
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </main>
    );
}
