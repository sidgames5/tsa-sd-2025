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
            name: "Powdery Mildew",
            description: "Powdery mildew is a fungal disease that appears as a white, dusty coating on leaves and stems. It thrives in warm, humid conditions, weakening plants and reducing yields if left untreated. It spreads rapidly, impacting surrounding plants.",
            appearance: " - White or grayish-white powdery growth on leaves and stems\n - Leaves may appear twisted, curled, or exhibit yellow spots.",
            fix: " - Use a fungicide (like chlorothalonil) following product label instructions\n - Remove affected leaves immediately\n - Ensure adequate sunlight exposure\n - Water at the base of plants to avoid wetting foliage."
        },
        {
            name: "Chlorosis (Yellowing Leaves)",
            description: "Chlorosis, or yellowing leaves, indicates a deficiency in chlorophyll. This can be caused by improper watering, nutrient deficiencies (nitrogen, iron, magnesium), or environmental stressors, impacting plant health and productivity.",
            appearance: " - Leaves turn yellow, starting with older leaves, potentially progressing to brown and brittle\n - Leaves may appear dry and shriveled, with veins remaining green.",
            fix: " - Adjust watering practices based on soil moisture\n - Improve soil drainage to prevent waterlogging\n - Ensure plants receive sufficient sunlight\n - Use a balanced fertilizer with iron, nitrogen, and magnesium."
        },
        {
            name: "Spots on Leaves",
            description: "Leaf spots are caused by various fungal or bacterial pathogens, causing localized tissue damage and discoloration. Common examples include downy mildew, anthracnose, and late blight, each with distinct symptoms.",
            appearance: "Downy Mildew: Yellow spots on upper leaf surfaces\nAnthracnose: Sunken lesions, spots, and holes on leaves and stems\nLeaf Spots: Small, circular, dark brown or black spots\nLate Blight: Dark spots on leaves and brown lesions on stems.",
            fix: " - Remove infected leaves to prevent spread\n - Apply fungicides as needed, following product instructions\n - Avoid overhead watering to minimize leaf moisture\n - Ensure good air circulation around plants."
        },
        {
            name: "Pest Infestation",
            description: "Pest infestations, such as aphids, spider mites, and whiteflies, damage plants by extracting sap. This weakens plants, reduces yields, and can transmit diseases.",
            appearance: " - Leaves may yellow, brown, become sticky, or exhibit small holes\n - Visible insects or webbing on leaves.",
            fix: " - Apply neem oil or insecticidal soap according to instructions\n - Introduce beneficial insects for natural pest controls\n - Regularly inspect plants for pests\n - Maintain plant hygiene to deter pests."
        },
        {
            name: "Scab/Rot",
            description: "A disease caused by a fungus, leading to dark, scabby lesions on leaves and fruit, causing premature leaf drop, fruit deformation or tree death.",
            appearance: "Olive-green black, or purple, velvety spots on leaves and fruit which can later turn brown and corky or enlarge.",
            fix: " - Remove and dispose of fallen leaves in the fall to reduce the fungal inoculum\n - Apply fungicides (i.e. Dalicon) to protect healthy trees/vine from infection.\n - Select apple and crabapple varieties that are resistant to apple scab\n - Prune trees/vines to improve air circulation and reduce humidity"
        },
        {
            name: "Corn - Northern Leaf Blight",
            description: "Northern corn leaf blight is caused by a fungus, which manifests in tan or grayish lesions on leaves. ",
            appearance: " - Tan, grayish, oblong lesions on leaf surfaces\n - Spores produced on the underside of the leaf give the appearance of a dusty green fuzz",
            fix: " - Plant resistant corn hybrids\n - Reducing corn residue can also help to minimize the amount of inoculum"
        },
        {
            name: "Bacterial spots",
            description: "Bacterial spots are caused by various bacteria, leading to dark, water-soaked lesions on leaves and stems. This can cause leaf drop and reduced plant vigor.",
            appearance: "Dark brown spots on leaves",
            fix: " - Remove infected leaves and stems\n - Apply copper-based bactericides as needed\n - Avoid overhead watering to minimize leaf moisture\n - Ensure good air circulation around plants."
            
        },
        {
            name: "Blight",
            description: "Blight is a rapid and extensive decay of plant tissues, often caused by fungal or bacterial pathogens. It can affect leaves, stems, and fruits, leading to wilting and death.",
            appearance: " - Rapid wilting and browning of leaves\n - Black or brown lesions on stems and fruits.",
            fix: " - Remove infected plant parts immediately\n - Apply fungicides or bactericides as needed\n - Ensure proper watering practices to avoid excess moisture\n - Rotate crops to prevent disease recurrence."
        },
        {
            name: "Root Rot",
            description: "Root rot is a fungal disease caused by overwatering or poor drainage, leading to root decay and plant wilting. Symptoms include yellowing leaves, stunted growth, and mushy roots.",
            appearance: " - Yellowing leaves\n - Wilting or drooping\n - Mushy or blackened roots.",
            fix: " - Improve soil drainage\n - Reduce watering frequency\n - Use fungicides if necessary\n - Remove affected plants to prevent spread."
        },
        {
            name: "Fungal Infections",
            description: "Fungal infections can cause various symptoms, including leaf spots, wilting, and fruit rot. Common fungal diseases include powdery mildew, downy mildew, and rust.",
            appearance: " - Leaf spots or blights\n - Powdery or fuzzy growth on leaves\n - Wilting or yellowing leaves.",
            fix: " - Apply fungicides as needed\n - Remove infected plant parts\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Fungal Leaf Spot",
            description: "Fungal leaf spots are caused by various fungi, leading to dark, circular spots on leaves. They can cause leaf drop and reduce plant vigor.",
            appearance: " - Dark, circular spots on leaves\n - Yellow halos around spots.",
            fix: " - Remove infected leaves\n - Apply fungicides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Fungal Wilt",
            description: "Fungal wilt is caused by soil-borne fungi that block water transport in plants, leading to wilting and death. Symptoms include yellowing leaves and stunted growth.",
            appearance: " - Wilting leaves\n - Yellowing or browning of lower leaves\n - Stunted growth.",
            fix: " - Rotate crops to prevent recurrence\n - Use resistant varieties if available\n - Improve soil drainage\n - Apply fungicides if necessary."
        },
        {
            name: "Leaf Curl",
            description: "Leaf curl is a condition where leaves become distorted or curled due to environmental stress, pests, or diseases. It can affect plant growth and yield.",
            appearance: " - Leaves may curl upwards or downwards\n - Distorted leaf shape.",
            fix: " - Identify and address the underlying cause (pests, diseases, environmental stress)\n - Ensure proper watering and fertilization practices\n - Prune affected leaves if necessary."
        },
        {
            name: "Leaf Blight",
            description: "Leaf blight is a condition where leaves develop large, dark lesions, often caused by fungal or bacterial pathogens. It can lead to leaf drop and reduced plant vigor.",
            appearance: " - Large, dark lesions on leaves\n - Yellowing or browning of leaf edges.",
            fix: " - Remove infected leaves\n - Apply fungicides or bactericides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Leaf Spot",
            description: "Leaf spot is a common plant disease characterized by small, round spots on leaves, caused by various fungi or bacteria. It can lead to leaf drop and reduced plant vigor.",
            appearance: " - Small, round spots on leaves\n - Yellow halos around spots.",
            fix: " - Remove infected leaves\n - Apply fungicides or bactericides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Wilting",
            description: "Wilting is a condition where plants droop due to lack of water or root damage. It can be caused by overwatering, underwatering, or root rot.",
            appearance: " - Drooping or wilting leaves and stems\n - Yellowing leaves.",
            fix: " - Adjust watering practices based on soil moisture\n - Improve soil drainage if necessary\n - Use fungicides if root rot is suspected."
        },
        {
            name: "Leaf Scorch",
            description: "Leaf scorch is a condition where leaf edges turn brown and crispy due to environmental stress, such as drought or excessive heat. It can affect plant health and appearance.",
            appearance: " - Brown, crispy leaf edges\n - Yellowing or browning of leaf tips.",
            fix: " - Ensure proper watering practices\n - Provide shade during extreme heat\n - Avoid fertilizing during drought conditions."
        },
        {
            name: "Root Damage",
            description: "Root damage can occur due to overwatering, underwatering, or physical injury. It can lead to wilting, yellowing leaves, and stunted growth.",
            appearance: " - Wilting or drooping leaves\n - Yellowing leaves\n - Stunted growth.",
            fix: " - Adjust watering practices based on soil moisture\n - Improve soil drainage if necessary\n - Avoid physical damage to roots during planting or transplanting."
        },
        {
            name: "Leaf Drop",
            description: "Leaf drop is a condition where leaves fall off plants prematurely, often due to environmental stress, pests, or diseases. It can affect plant health and appearance.",
            appearance: " - Leaves may turn yellow or brown before dropping\n - Premature leaf drop from stems.",
            fix: " - Identify and address the underlying cause (pests, diseases, environmental stress)\n - Ensure proper watering and fertilization practices\n - Prune affected leaves if necessary."
        },
        {
            name: "Fruit Rot",
            description: "Fruit rot is a condition where fruits develop soft, mushy spots due to fungal or bacterial infections. It can lead to reduced yields and poor fruit quality.",
            appearance: " - Soft, mushy spots on fruits\n - Discoloration or mold growth.",
            fix: " - Remove infected fruits immediately\n - Apply fungicides or bactericides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Leaf Necrosis",
            description: "Leaf necrosis is a condition where leaf tissue dies, leading to brown or black spots. It can be caused by environmental stress, pests, or diseases.",
            appearance: " - Brown or black spots on leaves\n - Yellowing or browning of leaf edges.",
            fix: " - Identify and address the underlying cause (pests, diseases, environmental stress)\n - Ensure proper watering and fertilization practices\n - Prune affected leaves if necessary."
        },
        {
            name: "Leaf Distortion",
            description: "Leaf distortion is a condition where leaves become twisted or curled due to environmental stress, pests, or diseases. It can affect plant growth and yield.",
            appearance: " - Twisted or curled leaves\n - Distorted leaf shape.",
            fix: " - Identify and address the underlying cause (pests, diseases, environmental stress)\n - Ensure proper watering and fertilization practices\n - Prune affected leaves if necessary."
        },
        {
            name: "Fruit Scab",
            description: "Fruit scab is a fungal disease that causes dark, scabby lesions on fruits, reducing their quality and marketability. It can affect various fruit crops.",
            appearance: " - Dark, scabby lesions on fruits\n - Deformed or misshapen fruits.",
            fix: " - Remove and dispose of infected fruits\n - Apply fungicides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Fruit Sunburn",
            description: "Fruit sunburn occurs when fruits are exposed to excessive sunlight, leading to sunscald and discoloration. It can affect fruit quality and marketability.",
            appearance: " - Sunken, discolored areas on fruits\n - Scorched or sunburned appearance.",
            fix: " - Provide shade during extreme heat\n - Use reflective mulch to protect fruits\n - Ensure proper watering practices."
        },
        {
            name: "Fruit Blight",
            description: "Fruit blight is a condition where fruits develop dark, mushy spots due to fungal or bacterial infections. It can lead to reduced yields and poor fruit quality.",
            appearance: " - Dark, mushy spots on fruits\n - Discoloration or mold growth.",
            fix: " - Remove infected fruits immediately\n - Apply fungicides or bactericides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Fruit Drop",
            description: "Fruit drop is a condition where fruits fall off plants prematurely, often due to environmental stress, pests, or diseases. It can affect plant health and yield.",
            appearance: " - Fruits may turn yellow or brown before dropping\n - Premature fruit drop from stems.",
            fix: " - Identify and address the underlying cause (pests, diseases, environmental stress)\n - Ensure proper watering and fertilization practices\n - Prune affected fruits if necessary."
        },
        {
            name: "Leaf Spotting",
            description: "Leaf spotting is a condition where leaves develop small, dark spots due to fungal or bacterial infections. It can lead to leaf drop and reduced plant vigor.",
            appearance: " - Small, dark spots on leaves\n - Yellow halos around spots.",
            fix: " - Remove infected leaves\n - Apply fungicides or bactericides as needed\n - Ensure good air circulation around plants\n - Avoid overhead watering."
        },
        {
            name: "Leaf Browning",
            description: "Leaf browning is a condition where leaf edges turn brown and crispy due to environmental stress, such as drought or excessive heat. It can affect plant health and appearance.",
            appearance: " - Brown, crispy leaf edges\n - Yellowing or browning of leaf tips.",
            fix: " - Ensure proper watering practices\n - Provide shade during extreme heat\n - Avoid fertilizing during drought conditions."
        },
        {
            name: "Leaf Wilting",
            description: "Leaf wilting is a condition where leaves droop due to lack of water or root damage. It can be caused by overwatering, underwatering, or root rot.",
            appearance: " - Drooping or wilting leaves and stems\n - Yellowing leaves.",
            fix: " - Adjust watering practices based on soil moisture\n - Improve soil drainage if necessary\n - Use fungicides if root rot is suspected."
        },
        {
            name: "Leaf Yellowing",
            description: "Leaf yellowing is a condition where leaves turn yellow due to nutrient deficiencies, environmental stress, or diseases. It can affect plant health and yield.",
            appearance: " - Yellowing leaves, starting with older leaves\n - Leaves may appear dry and shriveled.",
            fix: " - Adjust watering practices based on soil moisture\n - Improve soil drainage to prevent waterlogging\n - Ensure plants receive sufficient sunlight\n - Use a balanced fertilizer with iron, nitrogen, and magnesium."
        }
    ];

    const [searchTerm, setSearchTerm] = useState('');

    const filteredDiseases = diseases.filter(diseases => 
        diseases.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className={`${cookies.darkMode ? "bg-gray-900 text-white" : "bg-stone-50 text-black"} w-full flex flex-col items-center justify-center`}>
            <h1 className="text-5xl font-bold text-sky-600">Plant Diagnosis Guide</h1>
            <p className="text-xl mt-4">Match the symptoms with treatments for your plants.</p>
            <p className="text-xl">Click on each box for more information!</p>

            <div className="p-8">
                <input type="text"
                       placeholder='Search Diseases...'
                       value={searchTerm}
                       onChange={(e) => setSearchTerm(e.target.value)}
                       className="w-[75vw] p-2 border rounded text-black"
                />
            </div>
            
    

            {/* Disease Cards with Scroll Effect */}
            <motion.div
                // style={{ maskImage }}
                className="overflow-y-scroll flex flex-wrap gap-4 items-center justify-center mt-7 px-4 h-[90vh] w-[80vw] mb-8"
            >
                {/* from-gray-200 to-orange-300 */}
                {filteredDiseases.map((disease, index) => (
                    <motion.div key={index}
                        className={`bg-gradient-to-r max-w-[20vw] h-[10vh] p-5 rounded-lg cursor-pointer flex-shrink-0 border-[3px] border-gray-950 ${cookies.darkMode ? "from-sky-700 to-blue-900" : "from-sky-100 to-blue-300 text-gray-700"}`}
                        whileHover={{ scale: 1.35 }}
                        transition={{ type: "spring", stiffness: 100 }}
                        onClick={() => {
                            setSelectedDisease(disease);
                            setIsModalOpen(true);
                        }}
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.45 }}
                    >
                        <h3 className="font-bold">{disease.name}</h3>
                    </motion.div>
                ))}
            </motion.div>

            {/* Disease Details Modal */}
            {isModalOpen && selectedDisease && (
                <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", damping: 10 }}
                    className={`fixed max-w-[55vw] max-h-[70vh] w-fit h-fit top-1/4 left-1/4 transform -translate-x-1/2 -translate-y-1/2 text-white bg-gradient-to-r ${cookies.darkMode ? "from-gray-950 to-sky-950 border-white" : "from-gray-100 to-gray-200 border-black"} shadow-lg rounded-lg p-6 z-50 border-4`}
                >
                    <button
                        onClick={() => setIsModalOpen(false)}
                        className="m-2 bg-gray-500 text-white p-4 rounded-full absolute -left-5 -translate-x-full -translate-y-full w-4 h-4 flex flex-wrap items-center align-middle justify-center"
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
                                <td className="font-bold p-4 bg-amber-50 text-black rounded-tl-xl">How to Identify</td>
                                <td className="p-4 w-[50vw] bg-white text-left text-black rounded-tr-xl">
                                    {selectedDisease.appearance.split('\n').map((line, idx) => (
                                        <span key={idx}>{line}<br /></span>
                                    ))}
                                </td>
                            </tr>
                            <tr>
                                <td className="font-bold p-4 bg-amber-50 text-black rounded-bl-xl">How to Fix</td>
                                <td className="p-4 w-[50vw] bg-white text-left text-black rounded-br-xl">
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
