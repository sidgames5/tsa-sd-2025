"use client";

import {
    animate,
    motion,
    useMotionValue,
    useMotionValueEvent,
    useScroll
} from "framer-motion";
import { useRef, useState } from "react";
import { useCookies } from "react-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faClose } from "@fortawesome/free-solid-svg-icons";

export default function DiagnosisPage() {
    const ref = useRef(null);
    const { scrollXProgress } = useScroll({ container: ref });
    const maskImage = useScrollOverflowMask(scrollXProgress);

    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedDisease, setSelectedDisease] = useState(null);

    const [cookies] = useCookies(["darkMode"]);

    const diseases = [
        {
            name: "Powdery Leaves",
            solution: " - Remove affected leaves\n - Improve air circulation\n - Apply a fungicide",
            description: "Powdery mildew is a common fungal disease. The fungus manifest as a white, opwdery growth on the surface of leaves and stems, leading to leaf disotrtion and yellowing, which can reduce yields. ",
            appearance: " - White, powdery, grayish-white talcum-powder like growth on leaves\n - Twisted or distorted leaves",
            fix: "- Use fungicides containing chlorothalonil\n- Remove affected leaves"
        },
        {
            name: "Yellowing Leaves/Chlorosis",
            solution: " - Check for overwatering\n - Ensure proper drainage\n - Provide adequate sunlight",
            description: "Yellowing leaves may be a sign of overwatering, underwateringnutrient deficiency (nitrogen, iron, or magnesium), pests, temperature or poor drainage.",
            appearance: " - Leaves turn yellow and slowly turn brown\n - Dead shriveled leaves",
            fix: "- Check for overwatering\n- Ensure proper soil drainage\n- Provide adequate sunlight"
        },
        {
            name: "Wilting Plant",
            solution: " - Water if the soil is dry\n - Check for root rot",
            description: "Wilting plants may indicate dehydration, which can be exaggerated by factors such as extreme temperature, wind, and poor soil.\nLikewise, too much",
            appearance: " - Leaves turn yellow and slowly turn brown\n - Dead shriveled leaves",
            fix: "- Water plants if soil is dry\n- Inspect for root rot\n- Improve soil aeration"
        },
        {
            name: "Spots on Leaves",
            solution: " - Could indicate fungal infection\n - Remove infected leaves\n - Avoid overhead watering",
            description: "Leaf spots may indicate powdery mildew (See Powdery Leaves section), downy mildew, anthracnose, leaf spots, leaf rust, and late blight.",
            appearance: "Downy Mildew: Yellowing on top of leaves\nAnthracnose: Characterized by sunken lesions, spots, andholes on leaves and stems\nLeaf Spots: Small, circular spots that are dark brown\nLate Blight: Affects tomatoes and potatoes, dark spots on leaves and brown lesions in stems",
            fix: "- Remove infected leaves\n- Apply fungicide if needed\n- Avoid overhead watering"
        },
        {
            name: "Pests on Leaves",
            solution: " - Spray with neem oil or insecticidal soap to control pests",
            description: "Common plant pests include aphids, spider mites, and whiteflies, which damage leaves by sucking sap.",
            appearance: " - Leaves turn yellow and slowly turn brown\n - Dead shriveled leaves",
            fix: "- Spray neem oil or insecticidal soap\n- Introduce beneficial insects like ladybugs"
        },
        {
            name: "Drooping Leaves",
            solution: " - Check for underwatering or excessive heat exposure\n - Adjust watering schedule",
            description: "Drooping leaves are often a sign of underwatering, excessive heat, or root damage.",
            appearance: " - Leaves turn yellow and slowly turn brown\n - Dead shriveled leaves",
            fix: "- Water the plant properly\n- Ensure protection from extreme heat\n- Check for root damage"
        }
    ];

    return (
        <div className={`${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"} w-full h-full flex flex-col items-center align-middle justify-center`}>
            <h1 className="text-5xl font-bold">Plant Diagnosis Guide</h1>
            <p className="text-xl mt-4">Match the symptoms with treatments for your plants.</p>

            {/* Scroll-linked progress bar */}
            <svg id="progress" width="80" height="80" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="30" pathLength="1" className="bg" />
                <motion.circle
                    cx="50"
                    cy="50"
                    r="30"
                    className="indicator"
                    style={{ pathLength: scrollXProgress }}
                />
            </svg>

            {/* Disease Cards with Scroll Effect */}
            <motion.div
                ref={ref}
                style={{ maskImage }}
                className="overflow-x-auto flex flex-nowrap gap-4 px-4 w-[80vw] max-w-[1200px] h-[30vh]"
            >

                {diseases.map((disease, index) => (
                    <motion.div key={index}
                        className="mt-[5vh] bg-gradient-to-r from-gray-500 to-gray-700 w-[20vw] h-[20vh] p-5 rounded-lg cursor-pointer flex-shrink-0 border-[3px] border-blue-700"
                        whileHover={{ scale: 1.05 }}
                        transition={{ type: "spring", stiffness: 100 }}
                        onClick={() => {
                            setSelectedDisease(disease);
                            setIsModalOpen(true);
                        }}
                    >
                        <h3 className="font-bold">{disease.name}</h3>
                        <p>
                            {disease.solution.split('\n').map((line, idx) => (
                                <span key={idx}>{line}<br /></span>
                            ))}
                        </p>
                    </motion.div>
                ))}
            </motion.div>

            {/* Disease Details Modal */}
            {isModalOpen && selectedDisease && (
                <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", damping: 10 }}
                    className="fixed max-w-[50vw] max-h-[60vh] w-fit h-fit top-1/4 left-1/4 transform -translate-x-1/2 -translate-y-1/2 text-white bg-gradient-to-tr from-gray-500 to-gray-700 shadow-lg rounded-lg p-6 z-50 border-4 border-blue-500"
                >
                    <h1 className="text-3xl font-bold">{selectedDisease.name} Overview</h1>
                    <p className="text-lg text-left">{selectedDisease.description}</p>

                    <table className="mt-5">
                        <tbody>
                            <tr>
                                <td className="font-bold p-4 bg-gray-100 text-black rounded-tl-xl">How to Identify</td>
                                <td className="p-4 w-[50vw] bg-gray-200 text-left text-black rounded-tr-xl">
                                    {selectedDisease.appearance.split('\n').map((line, idx) => (
                                        <span key={idx}>{line}<br /></span>
                                    ))}
                                </td>
                            </tr>
                            <tr>
                                <td className="font-bold p-4 bg-gray-100 text-black rounded-bl-xl">How to Fix</td>
                                <td className="p-4 w-[50vw] bg-gray-200 text-left text-black rounded-br-xl">
                                    {selectedDisease.fix.split('\n').map((line, idx) => (
                                        <span key={idx}>{line}<br /></span>
                                    ))}
                                </td>
                            </tr>
                        </tbody>
                    </table>

                    <button
                        onClick={() => setIsModalOpen(false)}
                        className="m-2 bg-gray-500 text-white p-4 rounded-full absolute -left-4 -top-4 -translate-x-full -translate-y-full w-4 h-4 flex flex-row items-center align-middle justify-center"
                    >
                        <motion.span whileHover={{ scale: 1.2 }} className="w-full h-full flex flex-row items-center justify-center align-middle">
                            <FontAwesomeIcon icon={faClose} />
                        </motion.span>
                    </button>
                </motion.div>
            )}

            <StyleSheet />
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
        } else {
            animate(maskImage, `linear-gradient(90deg, transparent, #000 20%, #000 80%, transparent)`);
        }
    });

    return maskImage;
}

// Styles
function StyleSheet() {
    return (
        <style>{`
            #progress {
                position: absolute;
                top: -65px;
                left: -15px;
                transform: rotate(-90deg);
            }

            .bg { stroke: var(--layer); }

            #progress circle {
                stroke-dashoffset: 0;
                stroke-width: 10%;
                fill: none;
            }

            .indicator { stroke: var(--accent); }

            .overflow-x-scroll {
                display: flex;
                list-style: none;
                padding: 20px 0;
                margin: 0 auto;
                gap: 20px;
            }

            ::-webkit-scrollbar {
                height: 5px;
                width: 5px;
                background: #fff3;
            }

            ::-webkit-scrollbar-thumb {
                background: var(--accent);
            }
        `}</style>
    );
}
