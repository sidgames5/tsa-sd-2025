"use client";

import { animate, motion, useMotionValue, useMotionValueEvent, useScroll } from "framer-motion";
import { useRef, useState } from "react";
import "./DiagnosisPage.css";

export default function DiagnosisPage() {
    const ref = useRef(null);
    const { scrollXProgress } = useScroll({ container: ref });
    const maskImage = useScrollOverflowMask(scrollXProgress);

    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedDisease, setSelectedDisease] = useState(null);

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
        <div className="container">
            <h1 className="header">Plant Diagnosis Guide</h1>
            <p className="subtext">Match the symptoms with treatments for your plants.</p>

            <motion.ul ref={ref} style={{ maskImage }} className="disease-list overflow-visible">
                {diseases.map((disease, index) => (
                    <div key={index} onClick={() => {
                        setSelectedDisease(disease);
                        setIsModalOpen(true);
                    }}>
                        <motion.li
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

            {isModalOpen && selectedDisease && (
                <motion.div id="details-modal" className="fixed max-w-[50vw] max-h-[60vh] w-fit h-fit top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white bg-gradient-to-tr to-blue-950 from-green-950 shadow-lg rounded-lg p-6 z-50">
                    <h1 className="text-nowrap text-3xl">{selectedDisease.name} Overview</h1>
                    <p className="text-lg text-left text-white">
                        {selectedDisease.description}
                    </p>

                    <table className="mt-5">
                        <tbody>
                            <tr>
                                <td className="font-bold text-1xl p-4 bg-lime-100 rounded-tl-xl text-black">How to Identify</td>
                                <td className="text-1xl p-4 w-[50vw] bg-gray-100 rounded-tr-xl text-left text-black">
                                    {selectedDisease.appearance.split('\n').map((line, idx) => (
                                        <span key={idx}>
                                            {line}
                                            <br />
                                        </span>
                                    ))}
                                </td>
                            </tr>
                            <tr>
                                <td className="font-bold text-1xl p-4 bg-lime-100 rounded-bl-xl text-black">How to Fix</td>
                                <td className="text-1xl p-4 w-[50vw] bg-gray-100 rounded-br-xl text-left text-black">
                                    {selectedDisease.fix.split('\n').map((line, idx) => (
                                        <span key={idx}>
                                            {line}
                                            <br />
                                        </span>
                                    ))}
                                </td>
                            </tr>
                        </tbody>
                    </table>

                    <button
                        onClick={() => setIsModalOpen(false)}
                        className="mt-4 bg-red-500 text-white px-4 py-2 rounded"
                    >
                        Close
                    </button>
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
        } else if (scrollXProgress.getPrevious() === 0 || scrollXProgress.getPrevious() === 1) {
            animate(maskImage, `linear-gradient(90deg, transparent, #000 20%, #000 80%, transparent)`);
        }
    });

    return maskImage;
}
