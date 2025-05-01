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
            const botMessage = { sender: "bot", text: formatBotResponse(data.reply) };
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

    const formatBotResponse = (response) => {
        // Convert bullet points into HTML
        return response
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // Bold text
            .replace(/\* (.*?)\n/g, "<ul><li>$1</li></ul>")  // Bullet points
            .replace(/\n/g, "<br />"); // Line breaks
    };

    return (
        <main className={`w-full min-h-screen py-20 px-6 bg-gradient-to-r ${isDark ? "from-purple-600 via-pink-600 to-red-500" : "from-teal-500 to-yellow-500"}`}>
            <div className="max-w-3xl mx-auto">
                <h1 className="text-4xl font-semibold text-center mb-10 drop-shadow-lg text-white">
                    LeafLogic Support Assistant
                </h1>
                
                {/* Messages Container */}
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 space-y-4 max-h-[70vh] overflow-y-auto mb-6">
                    {messages.map((msg, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className={`rounded-xl px-6 py-4 w-fit max-w-[80%] ${
                                msg.sender === "user" 
                                    ? "ml-auto bg-gradient-to-r from-blue-500 to-blue-600 text-white" 
                                    : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-100"
                            }`}
                            dangerouslySetInnerHTML={{ __html: msg.text }}
                        />
                    ))}
                    {loading && <div className="text-sm text-gray-500 dark:text-gray-400 italic">Thinking...</div>}
                </div>

                {/* Input Area */}
                <div className="flex gap-4 items-center">
                    <input
                        type="text"
                        placeholder="Ask a farming question..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className={`flex-grow px-4 py-3 rounded-lg border transition-colors ${
                            isDark 
                                ? 'border-gray-600 bg-gray-800 text-white placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500' 
                                : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:border-blue-500 focus:ring-2 focus:ring-blue-500'
                        }`}
                        onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    />
                    <button
                        onClick={handleSend}
                        className="p-3 bg-gradient-to-r from-blue-600 to-blue-800 text-white rounded-lg transition-transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
                    >
                        <Send size={22} />
                    </button>
                </div>
            </div>
        </main>
    );
}