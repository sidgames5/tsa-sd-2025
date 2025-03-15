"use client";

import { animate, motion, useMotionValue, useMotionValueEvent, useScroll } from "framer-motion";
import { useRef, useState } from "react";
import "./DiagnosisPage.css";
import PowderyModal from "./PowderyModal";
import YellowingModal from "./Yellowing";

export default function DiagnosisPage() {
    const ref = useRef(null);
    const { scrollXProgress } = useScroll({ container: ref });
    const maskImage = useScrollOverflowMask(scrollXProgress);
    const [isModalOpen, setIsModalOpen] = useState(false);

    const diseases = [
        { name: "Powdery Leaves:", solution: " - Remove affected leaves\n - Improve air circulation\n - Apply a fungicide" },
        { name: "Yellowing Leaves:", solution: " - Check for overwatering\n - Ensure proper drainage\n - Provide adequate sunlight" },
        { name: "Wilting Plant:", solution: " - Water if the soil is dry\n - Check for root rot" },
        { name: "Spots on Leaves:", solution: " - Could indicate fungal infection\n - Remove infected leaves\n - Avoid overhead watering" },
        { name: "Pests on Leaves:", solution: " - Spray with neem oil or insecticidal soap to control pests" },
        { name: "Drooping Leaves:", solution: " - Check for underwatering or excessive heat exposure\n - Adjust watering schedule" },
    ];

    return (
        <div className="container">
            <h1 className="header">Plant Diagnosis Guide</h1>
            <p className="subtext">Match the symptoms with treatments for your plants.</p>

            <motion.ul ref={ref} style={{ maskImage }} className="disease-list overflow-visible">
                {diseases.map((disease, index) => (
                    <div onClick={() => { setIsModalOpen(true) }}>
                        <motion.li
                            key={index}
                            className="disease-item overflow-visible cursor-pointer"
                            whileHover={{ scale: 1.15 }}
                            transition={{ type: "spring", stiffness: 50 }}
                        >
                            <h3>{disease.name}</h3>
                            <p>
                                {disease.solution.split('\n').map((line, idx) => (
                                    <span key={idx}>
                                        {line}
                                        <br />
                                    </span>
                                ))}
                            </p>
                        </motion.li>
                    </div>
                ))}
            </motion.ul>
            <div id="details-modal" className={`fixed ${isModalOpen ? "block" : "hidden"} max-w-[50vw] max-h-[60vh] w-fit h-fit top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gradient-to-tr to-blue-200 from-green-200 shadow-lg rounded-lg p-6 z-50`}>
                <PowderyModal onClose={() => { setIsModalOpen(false) }}>
                    <h1 className="text-nowrap text-3xl">Powdery Plants Overview</h1>
                    <p className=" text-lg text-left font-serif">
                        Powdery leaves on crops, often appearing as a <span className="text-white">white</span> or <span className="text-gray-400">grayish</span> powdery coating, are a sign of powdery mildew. A fungal disease that affects a wide range of plants, including vegetables, fruits, and ornamentals.
                    </p>

                    <table className="mt-5">
                        <tbody>
                            <tr>
                                <td className="font-bold text-1xl p-4 bg-lime-100 rounded-tl-xl">How to Identify</td>
                                <td className="text-1xl p-4 w-[50vw] bg-gray-100 rounded-tr-xl">
                                    If there is a mass of powdery white fungus on you're crop's leaves, it has powdery mildew. This will cause the leaves to distort, yellow, and eventually die along with the plant.
                                </td>
                            </tr>
                            <tr>
                                <td className="font-bold text-1xl p-4 bg-lime-100 rounded-bl-xl">How to Fix</td>
                                <td className="text-1xl p-4 w-[50vw] bg-gray-100 rounded-br-xl text-left">
                                    - Use fungicides containing active ingredients like chlorothalonil
                                    <br />
                                    - Remove affected leaves
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </PowderyModal>
                YellowingModal

            </div>
        </div>
    );
}

// Function to create a smooth scroll effect
function useScrollOverflowMask(scrollXProgress) {
    const maskImage = useMotionValue(
        `linear-gradient(90deg, #000, #000 0%, #000 80%, transparent)`
    );

    useMotionValueEvent(scrollXProgress, "change", (value) => {
        if (value === 0) {
            animate(maskImage, `linear-gradient(90deg, #000, #000 0%, #000 80%, transparent)`);
        } else if (value === 1) {
            animate(maskImage, `linear-gradient(90deg, transparent, #000 20%, #000 100%, #000)`);
        } else if (scrollXProgress.getPrevious() === 0 || scrollXProgress.getPrevious() === 1) {
            animate(maskImage, `linear-gradient(90deg, transparent, #000 20%, #000 80%, transparent)`);
        }
    });

    return maskImage;
}
