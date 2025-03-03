"use client";

import { animate, motion, useMotionValue, useMotionValueEvent, useScroll } from "framer-motion";
import { useRef } from "react";
import "./DiagnosisPage.css";

export default function DiagnosisPage() {
    const ref = useRef(null);
    const { scrollXProgress } = useScroll({ container: ref });
    const maskImage = useScrollOverflowMask(scrollXProgress);

    const diseases = [
        { name: "Powdery Leaves:", solution: "Remove affected leaves\nImprove air circulation\nApply a fungicide." },
        { name: "Yellowing Leaves:", solution: "Check for overwatering\nEnsure proper drainage\nProvide adequate sunlight." },
        { name: "Wilting Plant:", solution: "Water if the soil is dry\nCheck for root rot." },
        { name: "Spots on Leaves:", solution: "Could indicate fungal infection\nRemove infected leaves\nAvoid overhead watering." },
        { name: "Pests on Leaves:", solution: "Spray with neem oil or insecticidal soap to control pests." },
        { name: "Drooping Leaves:", solution: "Check for underwatering or excessive heat exposure\nAdjust watering schedule." },
    ];

    return (
        <div className="container">
            <h1 className="header">Plant Diagnosis Guide</h1>
            <p className="subtext">Match the symptoms with treatments for your plants.</p>

            <motion.ul ref={ref} style={{ maskImage }} className="disease-list overflow-visible">
                {diseases.map((disease, index) => (
                    <motion.li
                        key={index}
                        className="disease-item overflow-visible"
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
                ))}
            </motion.ul>
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
