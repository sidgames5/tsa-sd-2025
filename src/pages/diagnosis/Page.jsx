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
            name: "Powdery Mildew (White, Powdery Leaves)",
            solution: " - Gently pluck off any leaves with the white powder.\n - Give your plants more space for better air flow.\n - Apply a fungicide spray following product instructions.\n - Mix 1 teaspoon of baking soda with 1 quart of water and a few drops of dish soap and spray the leaves.",
            description: "Powdery mildew is a fungal disease that appears as a white, dusty coating on leaves and stems. It thrives in warm, humid conditions, weakening plants and reducing yields if left untreated. It spreads rapidly, impacting surrounding plants.",
            appearance: " - White or grayish-white powdery growth on leaves and stems.\n - Leaves may appear twisted, curled, or exhibit yellow spots.",
            fix: " - Use a fungicide (like chlorothalonil) following product label instructions.\n - Remove affected leaves immediately.\n - Ensure adequate sunlight exposure.\n - Water at the base of plants to avoid wetting foliage."
        },
        {
            name: "Chlorosis (Yellowing Leaves)",
            solution: " - Check soil moisture to avoid over or under-watering.\n - Ensure proper soil drainage.\n - Provide adequate sunlight.\n - Apply a fertilizer with iron, nitrogen, and magnesium.",
            description: "Chlorosis, or yellowing leaves, indicates a deficiency in chlorophyll. This can be caused by improper watering, nutrient deficiencies (nitrogen, iron, magnesium), or environmental stressors, impacting plant health and productivity.",
            appearance: " - Leaves turn yellow, starting with older leaves, potentially progressing to brown and brittle.\n - Leaves may appear dry and shriveled, with veins remaining green.",
            fix: " - Adjust watering practices based on soil moisture.\n - Improve soil drainage to prevent waterlogging.\n - Ensure plants receive sufficient sunlight.\n - Use a balanced fertilizer with iron, nitrogen, and magnesium."
        },
        {
            name: "Wilting Plant (Droopy and Sad)",
            solution: " - Water plants if soil is dry.\n - Inspect roots for signs of root rot.\n - Loosen compacted soil to improve aeration.\n - Move plants to cooler locations during heat stress.",
            description: "Wilting occurs when plants lose turgidity due to water stress, heat, or root damage. This compromises plant health and can lead to significant yield reductions.",
            appearance: " - Leaves become floppy and may turn yellow or brown.\n - Plants exhibit a general loss of vigor and may lean or collapse.\n - Soil may be excessively dry or waterlogged.",
            fix: " - Water plants thoroughly when soil is dry.\n - Address root rot by removing affected roots and improving drainage.\n - Improve soil aeration to support root health.\n - Provide shade or relocate plants during extreme heat."
        },
        {
            name: "Spots on Leaves (Different Colors)",
            solution: " - Remove and dispose of affected leaves.\n - Water at the base of plants to avoid wetting foliage.\n - Apply a fungicide following product instructions.\n - Consider using copper-based fungicides.",
            description: "Leaf spots are caused by various fungal or bacterial pathogens, causing localized tissue damage and discoloration. Common examples include downy mildew, anthracnose, and late blight, each with distinct symptoms.",
            appearance: "Downy Mildew: Yellow spots on upper leaf surfaces.\nAnthracnose: Sunken lesions, spots, and holes on leaves and stems.\nLeaf Spots: Small, circular, dark brown or black spots.\nLate Blight: Dark spots on leaves and brown lesions on stems.",
            fix: " - Remove infected leaves to prevent spread.\n - Apply fungicides as needed, following product instructions.\n - Avoid overhead watering to minimize leaf moisture.\n - Ensure good air circulation around plants."
        },
        {
            name: "Pest Infestation (Tiny Bugs)",
            solution: " - Apply neem oil following product instructions.\n - Use insecticidal soap as directed.\n - Introduce beneficial insects like ladybugs.\n - Use a strong water spray to dislodge pests.",
            description: "Pest infestations, such as aphids, spider mites, and whiteflies, damage plants by extracting sap. This weakens plants, reduces yields, and can transmit diseases.",
            appearance: " - Leaves may yellow, brown, become sticky, or exhibit small holes.\n - Visible insects or webbing on leaves.",
            fix: " - Apply neem oil or insecticidal soap according to instructions.\n - Introduce beneficial insects for natural pest control.\n - Regularly inspect plants for pests.\n - Maintain plant hygiene to deter pests."
        },
        {
            name: "Drooping Leaves (Not Standing Upright)",
            solution: " - Check soil moisture and temperature.\n - Adjust watering schedule as needed.\n - Increase humidity through misting or humidifiers.\n - Inspect roots for damage.",
            description: "Drooping leaves indicate water stress, heat exposure, or root damage. This compromises plant health and can significantly reduce yields.",
            appearance: " - Leaves appear limp and lack rigidity.\n - Leaves may turn yellow or brown.\n - Soil may be excessively dry or waterlogged.",
            fix: " - Water plants when soil is dry.\n - Provide shade or relocate plants during heat stress.\n - Address root damage by removing affected roots and improving drainage.\n - Increase humidity to reduce transpiration stress."
        }
    ];

    return (
        <div className={`${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gray-100 text-black"} w-screen h-screen flex flex-col items-center align-middle justify-center`}>
            <h1 className="text-5xl font-bold text-sky-600 underline">Plant Diagnosis Guide</h1>
            <p className="text-xl mt-4">Match the symptoms with treatments for your plants.</p>
            <p className="text-xl">Click on each box for more information!</p>


            {/* Disease Cards with Scroll Effect */}
            <motion.div
                style={{ maskImage }}
                className="overflow-x-scroll flex flex-nowrap gap-4 px-4 w-[80vw] max-w-[1200px] h-[47vh]"
            >

                {diseases.map((disease, index) => (
                    <motion.div key={index}
                        className={`mt-5 bg-gradient-to-r w-[20vw] h-[42vh] p-5 rounded-lg cursor-pointer flex-shrink-0 border-[3px] border-gray-950 ${cookies.darkMode ? "from-blue-900 to-gray-800" : "from-blue-100 to-sky-200"}`}
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
                    className={`fixed max-w-[50vw] max-h-[65vh] w-fit h-fit top-1/4 left-1/4 transform -translate-x-1/2 -translate-y-1/2 text-white bg-gradient-to-r ${cookies.darkMode ? "from-gray-950 to-sky-950 border-white" : "from-gray-100 to-gray-50 border-black"} shadow-lg rounded-lg p-6 z-50 border-4`}
                >
                    <button
                        onClick={() => setIsModalOpen(false)}
                        className="m-2 bg-gray-500 text-white p-4 rounded-full absolute -left-5 -translate-x-full -translate-y-full w-4 h-4 flex flex-row items-center align-middle justify-center"
                    >
                        <motion.span whileHover={{ scale: 1.2 }} className="w-full h-full flex flex-row items-center justify-center align-middle">
                            <FontAwesomeIcon icon={faClose} />
                        </motion.span>
                    </button>

                    <h1 className={`text-3xl font-bold p-3 ${cookies.darkMode ? "text-white" : "text-black"}`}>{selectedDisease.name}</h1>
                    <p className={`text-lg text-left rounded-xl p-2 mt-2 cursor-text ${cookies.darkMode ? "text-white" : "text-black"}`}>
                        {selectedDisease.description}
                    </p>
                    <table className="mt-5 mb-5">
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


                </motion.div>
            )}

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
