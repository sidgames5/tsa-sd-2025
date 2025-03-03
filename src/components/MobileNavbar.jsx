import { Link } from "react-router";
import NavbarButton from "./NavbarButton";
import NavbarItemsList from "./NavbarItemsList";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faClose } from "@fortawesome/free-solid-svg-icons";

export default function MobileNavbar({ onClose }) {
    return <div className="bg-gray-900 w-full flex flex-row">
        <div className="w-fit flex flex-col gap-2 *:mx-4">
            {NavbarItemsList.map((v) => <NavbarButton>
                <Link to={v[1]}>
                    {v[0]}
                </Link>
            </NavbarButton>)}
        </div>
        <div className="mt-8 text-white text-2xl cursor-pointer hover:text-3xl transition-all duration-300"><FontAwesomeIcon icon={faClose} onClick={onClose} /></div>
    </div>
}