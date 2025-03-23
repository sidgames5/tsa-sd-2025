import { useCookies } from "react-cookie";

export default function NavbarButton({ children }) {
    const [cookies] = useCookies(["darkMode"]);

    return <div
        className={`list-none text-2xl font-bold ${cookies.darkMode ? "bg-gray-900 text-gray-100 border-white" : "bg-white text-black border-black"} border-2 rounded-lg px-4 py-2 transition duration-300 hover:bg-white hover:text-gray-900 hover:shadow-blue-500/50 hover:translate-y-2`}
    >
        {children}
    </div>;
}