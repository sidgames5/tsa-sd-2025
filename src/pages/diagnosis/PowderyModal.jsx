import { faClose } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React from "react";

export default function PowderyModal({ isOpen, onClose, children }) {
    return <div className="flex flex-row gap-4">
        <FontAwesomeIcon icon={faClose} className="cursor-pointer text-2xl" onClick={onClose} />
        <div>
            {children}
        </div>
    </div>;
}