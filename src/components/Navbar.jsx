import { Link } from "react-router";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faLeaf, faMoon } from "@fortawesome/free-solid-svg-icons";
import { useCookies } from "react-cookie";

export default function Navbar() {
  const [cookies, setCookies] = useCookies(["darkMode"]);

  return (
    <div className="w-full">
        <nav className={`${cookies.darkMode ? "bg-gray-900 text-white": "bg-gray-100 text-black"} w-full p-4 flex items-center`}>
            <div className="flex-shrink-0">
            <a href="/" className="text-lg font-semibold mr-8 flex items-center">
                Plantwise
                <FontAwesomeIcon style={{ color: "green" }} icon={faLeaf} />
            </a>
            </div>
            <div className="flex-grow flex justify-center gap-10">
                <button>
                    <Link to="/upload" className={`${cookies.darkMode ? "hover:text-gray-300": "hover:text-blue-500"} mr-4`}>
                        Upload
                    </Link>
                </button>
                <button>
                    <Link to="/diagnosis" className={`${cookies.darkMode ? "hover:text-gray-300": "hover:text-blue-500"} mr-4`}>
                        Diagnosis
                    </Link>
                </button>
                <button>
                    <Link to="/result" className={`${cookies.darkMode ? "hover:text-gray-300": "hover:text-blue-500"} mr-4`}>
                        AI Chart
                    </Link>
                </button>
            </div>
            <div className="flex-shrink-0">
                <button
                    onClick={() => {
                    if (cookies.darkMode) {
                        setCookies("darkMode", false);
                    } else {
                        setCookies("darkMode", true);
                    }
                    }}
                    className={`${cookies.darkMode ? "text-white hover:text-blue-300": "text-black hover:text-blue-900 duration-300"} p-4`}
                >
                    <FontAwesomeIcon icon={faMoon} />
                </button>
            </div>
      </nav>
    </div>
  );
}