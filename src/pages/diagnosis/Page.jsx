"use client";

import { motion } from "framer-motion";
import { useRef, useState } from "react";
import { useCookies } from "react-cookie";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faClose, faSearch, faLeaf, faChevronRight, faArrowDown} from "@fortawesome/free-solid-svg-icons";
import { AnimatePresence } from "framer-motion";
import AIChatbot from "/Users/kaniskprakash/Documents/GitHub/tsa-sd-2025/src/components/SupportAI.jsx";



export default function DiagnosisPage() {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedDisease, setSelectedDisease] = useState(null);
    const [cookies] = useCookies(["darkMode"]);
    const [searchTerm, setSearchTerm] = useState('');
    const containerRef = useRef(null);
    const [openSection, setOpenSection] = useState(null);
    const toggleSection = (section) => {
    setOpenSection(openSection === section ? null : section);
    };
    const constraintsRef = useRef(null);
    const diseases = [
        {
            name: "Powdery Mildew",
            description: "Powdery mildew is a fungal disease that appears as a white, dusty coating on leaves and stems. It thrives in warm, humid conditions, weakening plants and reducing yields if left untreated. It spreads rapidly, impacting surrounding plants.",
            appearance: "White or grayish-white powdery growth on leaves and stems\nLeaves may appear twisted, curled, or exhibit yellow spots.",
            fix: "Use a fungicide (like chlorothalonil) following product label instructions\nRemove affected leaves immediately\nEnsure adequate sunlight exposure\nWater at the base of plants to avoid wetting foliage."
        },
        {
            name: "Chlorosis (Yellowing Leaves)",
            description: "Chlorosis, or yellowing leaves, indicates a deficiency in chlorophyll. This can be caused by improper watering, nutrient deficiencies (nitrogen, iron, magnesium), or environmental stressors, impacting plant health and productivity.",
            appearance: "Leaves turn yellow, starting with older leaves, potentially progressing to brown and brittle\nLeaves may appear dry and shriveled, with veins remaining green.",
            fix: "Adjust watering practices based on soil moisture\nImprove soil drainage to prevent waterlogging\nEnsure plants receive sufficient sunlight\nUse a balanced fertilizer with iron, nitrogen, and magnesium."
        },
        {
            name: "Spots on Leaves",
            description: "Leaf spots are caused by various fungal or bacterial pathogens, causing localized tissue damage and discoloration. Common examples include downy mildew, anthracnose, and late blight, each with distinct symptoms.",
            appearance: "Downy Mildew: Yellow spots on upper leaf surfaces\nAnthracnose: Sunken lesions, spots, and holes on leaves and stems\nLeaf Spots: Small, circular, dark brown or black spots\nLate Blight: Dark spots on leaves and brown lesions on stems.",
            fix: "Remove infected leaves to prevent spread\nApply fungicides as needed, following product instructions\nAvoid overhead watering to minimize leaf moisture\nEnsure good air circulation around plants."
        },
        {
            name: "Pest Infestation",
            description: "Pest infestations, such as aphids, spider mites, and whiteflies, damage plants by extracting sap. This weakens plants, reduces yields, and can transmit diseases.",
            appearance: "Leaves may yellow, brown, become sticky, or exhibit small holes\nVisible insects or webbing on leaves.",
            fix: "Apply neem oil or insecticidal soap according to instructions\nIntroduce beneficial insects for natural pest controls\nRegularly inspect plants for pests\nMaintain plant hygiene to deter pests."
        },
        {
            name: "Scab/Rot",
            description: "A disease caused by a fungus, leading to dark, scabby lesions on leaves and fruit, causing premature leaf drop, fruit deformation or tree death.",
            appearance: "Olive-green black, or purple, velvety spots on leaves and fruit which can later turn brown and corky or enlarge.",
            fix: "Remove and dispose of fallen leaves in the fall to reduce the fungal inoculum\nApply fungicides (i.e. Dalicon) to protect healthy trees/vine from infection.\nSelect apple and crabapple varieties that are resistant to apple scab\nPrune trees/vines to improve air circulation and reduce humidity"
        },
        {
            name: "Corn - Northern Leaf Blight",
            description: "Northern corn leaf blight is caused by a fungus, which manifests in tan or grayish lesions on leaves. ",
            appearance: "Tan, grayish, oblong lesions on leaf surfaces\nSpores produced on the underside of the leaf give the appearance of a dusty green fuzz",
            fix: "Plant resistant corn hybrids\nReducing corn residue can also help to minimize the amount of inoculum"
        },
        {
            name: "Bacterial spots",
            description: "Bacterial spots are caused by various bacteria, leading to dark, water-soaked lesions on leaves and stems. This can cause leaf drop and reduced plant vigor.",
            appearance: "Dark brown spots on leaves",
            fix: "Remove infected leaves and stems\nApply copper-based bactericides as needed\nAvoid overhead watering to minimize leaf moisture\nEnsure good air circulation around plants."
        },
        {
            name: "Blight",
            description: "Blight is a rapid and extensive decay of plant tissues, often caused by fungal or bacterial pathogens. It can affect leaves, stems, and fruits, leading to wilting and death.",
            appearance: "Rapid wilting and browning of leaves\nBlack or brown lesions on stems and fruits.",
            fix: "Remove infected plant parts immediately\nApply fungicides or bactericides as needed\nEnsure proper watering practices to avoid excess moisture\nRotate crops to prevent disease recurrence."
        },
        {
            name: "Root Rot",
            description: "Root rot is a fungal disease caused by overwatering or poor drainage, leading to root decay and plant wilting. Symptoms include yellowing leaves, stunted growth, and mushy roots.",
            appearance: "Yellowing leaves\nWilting or drooping\nMushy or blackened roots.",
            fix: "Improve soil drainage\nReduce watering frequency\nUse fungicides if necessary\nRemove affected plants to prevent spread."
        },
        {
            name: "Fungal Infections",
            description: "Fungal infections can cause various symptoms, including leaf spots, wilting, and fruit rot. Common fungal diseases include powdery mildew, downy mildew, and rust.",
            appearance: "Leaf spots or blights\nPowdery or fuzzy growth on leaves\nWilting or yellowing leaves.",
            fix: "Apply fungicides as needed\nRemove infected plant parts\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Fungal Leaf Spot",
            description: "Fungal leaf spots are caused by various fungi, leading to dark, circular spots on leaves. They can cause leaf drop and reduce plant vigor.",
            appearance: "Dark, circular spots on leaves\nYellow halos around spots.",
            fix: "Remove infected leaves\nApply fungicides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Fungal Wilt",
            description: "Fungal wilt is caused by soil-borne fungi that block water transport in plants, leading to wilting and death. Symptoms include yellowing leaves and stunted growth.",
            appearance: "Wilting leaves\nYellowing or browning of lower leaves\nStunted growth.",
            fix: "Rotate crops to prevent recurrence\nUse resistant varieties if available\nImprove soil drainage\nApply fungicides if necessary."
        },
        {
            name: "Leaf Curl",
            description: "Leaf curl is a condition where leaves become distorted or curled due to environmental stress, pests, or diseases. It can affect plant growth and yield.",
            appearance: "Leaves may curl upwards or downwards\nDistorted leaf shape.",
            fix: "Identify and address the underlying cause (pests, diseases, environmental stress)\nEnsure proper watering and fertilization practices\nPrune affected leaves if necessary."
        },
        {
            name: "Leaf Blight",
            description: "Leaf blight is a condition where leaves develop large, dark lesions, often caused by fungal or bacterial pathogens. It can lead to leaf drop and reduced plant vigor.",
            appearance: "Large, dark lesions on leaves\nYellowing or browning of leaf edges.",
            fix: "Remove infected leaves\nApply fungicides or bactericides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Leaf Spot",
            description: "Leaf spot is a common plant disease characterized by small, round spots on leaves, caused by various fungi or bacteria. It can lead to leaf drop and reduced plant vigor.",
            appearance: "Small, round spots on leaves\nYellow halos around spots.",
            fix: "Remove infected leaves\nApply fungicides or bactericides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Wilting",
            description: "Wilting is a condition where plants droop due to lack of water or root damage. It can be caused by overwatering, underwatering, or root rot.",
            appearance: "Drooping or wilting leaves and stems\nYellowing leaves.",
            fix: "Adjust watering practices based on soil moisture\nImprove soil drainage if necessary\nUse fungicides if root rot is suspected."
        },
        {
            name: "Leaf Scorch",
            description: "Leaf scorch is a condition where leaf edges turn brown and crispy due to environmental stress, such as drought or excessive heat. It can affect plant health and appearance.",
            appearance: "Brown, crispy leaf edges\nYellowing or browning of leaf tips.",
            fix: "Ensure proper watering practices\nProvide shade during extreme heat\nAvoid fertilizing during drought conditions."
        },
        {
            name: "Root Damage",
            description: "Root damage can occur due to overwatering, underwatering, or physical injury. It can lead to wilting, yellowing leaves, and stunted growth.",
            appearance: "Wilting or drooping leaves\nYellowing leaves\nStunted growth.",
            fix: "Adjust watering practices based on soil moisture\nImprove soil drainage if necessary\nAvoid physical damage to roots during planting or transplanting."
        },
        {
            name: "Leaf Drop",
            description: "Leaf drop is a condition where leaves fall off plants prematurely, often due to environmental stress, pests, or diseases. It can affect plant health and appearance.",
            appearance: "Leaves may turn yellow or brown before dropping\nPremature leaf drop from stems.",
            fix: "Identify and address the underlying cause (pests, diseases, environmental stress)\nEnsure proper watering and fertilization practices\nPrune affected leaves if necessary."
        },
        {
            name: "Fruit Rot",
            description: "Fruit rot is a condition where fruits develop soft, mushy spots due to fungal or bacterial infections. It can lead to reduced yields and poor fruit quality.",
            appearance: "Soft, mushy spots on fruits\nDiscoloration or mold growth.",
            fix: "Remove infected fruits immediately\nApply fungicides or bactericides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Leaf Necrosis",
            description: "Leaf necrosis is a condition where leaf tissue dies, leading to brown or black spots. It can be caused by environmental stress, pests, or diseases.",
            appearance: "Brown or black spots on leaves\nYellowing or browning of leaf edges.",
            fix: "Identify and address the underlying cause (pests, diseases, environmental stress)\nEnsure proper watering and fertilization practices\nPrune affected leaves if necessary."
        },
        {
            name: "Leaf Distortion",
            description: "Leaf distortion is a condition where leaves become twisted or curled due to environmental stress, pests, or diseases. It can affect plant growth and yield.",
            appearance: "Twisted or curled leaves\nDistorted leaf shape.",
            fix: "Identify and address the underlying cause (pests, diseases, environmental stress)\nEnsure proper watering and fertilization practices\nPrune affected leaves if necessary."
        },
        {
            name: "Fruit Scab",
            description: "Fruit scab is a fungal disease that causes dark, scabby lesions on fruits, reducing their quality and marketability. It can affect various fruit crops.",
            appearance: "Dark, scabby lesions on fruits\nDeformed or misshapen fruits.",
            fix: "Remove and dispose of infected fruits\nApply fungicides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Fruit Sunburn",
            description: "Fruit sunburn occurs when fruits are exposed to excessive sunlight, leading to sunscald and discoloration. It can affect fruit quality and marketability.",
            appearance: "Sunken, discolored areas on fruits\nScorched or sunburned appearance.",
            fix: "Provide shade during extreme heat\nUse reflective mulch to protect fruits\nEnsure proper watering practices."
        },
        {
            name: "Fruit Blight",
            description: "Fruit blight is a condition where fruits develop dark, mushy spots due to fungal or bacterial infections. It can lead to reduced yields and poor fruit quality.",
            appearance: "Dark, mushy spots on fruits\nDiscoloration or mold growth.",
            fix: "Remove infected fruits immediately\nApply fungicides or bactericides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Fruit Drop",
            description: "Fruit drop is a condition where fruits fall off plants prematurely, often due to environmental stress, pests, or diseases. It can affect plant health and yield.",
            appearance: "Fruits may turn yellow or brown before dropping\nPremature fruit drop from stems.",
            fix: "Identify and address the underlying cause (pests, diseases, environmental stress)\nEnsure proper watering and fertilization practices\nPrune affected fruits if necessary."
        },
        {
            name: "Leaf Spotting",
            description: "Leaf spotting is a condition where leaves develop small, dark spots due to fungal or bacterial infections. It can lead to leaf drop and reduced plant vigor.",
            appearance: "Small, dark spots on leaves\nYellow halos around spots.",
            fix: "Remove infected leaves\nApply fungicides or bactericides as needed\nEnsure good air circulation around plants\nAvoid overhead watering."
        },
        {
            name: "Leaf Browning",
            description: "Leaf browning is a condition where leaf edges turn brown and crispy due to environmental stress, such as drought or excessive heat. It can affect plant health and appearance.",
            appearance: "Brown, crispy leaf edges\nYellowing or browning of leaf tips.",
            fix: "Ensure proper watering practices\nProvide shade during extreme heat\nAvoid fertilizing during drought conditions."
        },
        {
            name: "Leaf Wilting",
            description: "Leaf wilting is a condition where leaves droop due to lack of water or root damage. It can be caused by overwatering, underwatering, or root rot.",
            appearance: "Drooping or wilting leaves and stems\nYellowing leaves.",
            fix: "Adjust watering practices based on soil moisture\nImprove soil drainage if necessary\nUse fungicides if root rot is suspected."
        },
        {
            name: "Leaf Yellowing",
            description: "Leaf yellowing is a condition where leaves turn yellow due to nutrient deficiencies, environmental stress, or diseases. It can affect plant health and yield.",
            appearance: "Yellowing leaves, starting with older leaves\nLeaves may appear dry and shriveled.",
            fix: "Adjust watering practices based on soil moisture\nImprove soil drainage to prevent waterlogging\nEnsure plants receive sufficient sunlight\nUse a balanced fertilizer with iron, nitrogen, and magnesium."
        },
        {
            name: "Esca",
            description: "Grapevine fungal disease (also known as black measles) that affect the woody parts of vines with black spots. It can reduce yields and even kill the vine",
            appearance: "Dark spotting\nStriped patterns on leaves\nDieback of shoots",
            fix: "Pruning affected areas\nLime sulfar sprays\nApplying fungicide"
        },
        {
            name: "Haunglongbing (HLB)",
            description: "Bacterial disease (also known as citrus greening) that affects all citrus plants. It can lead to significantly reduced yields and the death of the infected tree.",
            appearance: "Asymmetrical leaf blotching\nSmall, bitter fruit that remain green when ripe\nShoots with pale green/yellow growth\nStunted tree growth\nDieback of branches\nOff-season flowering",
            fix: "None"
        },
        {
            name: "Leaf Mold",
            description: "Fungal disease that affects plants in humid environments It creates a layer of mold on leaves that hinder photosynthesis and plant health. Severe infections can weaken the plant.",
            appearance: "White/grayich growth on leaves\nYellowing/browning of leaves\nStudnted growth in severe cases\nMay start as small spots",
            fix: "Improve air circulation around the plant\nAvoid overhead watering and keep leaves dry\nDispose of affected leaves\nApply a fungicide in severe cases"
        },
        {
            name: "Target Spot",
            description: "Fungal disease that affects vegetables and ornamentals. It is characterized by lesions on leaves that resemble a target/bullseye.",
            appearance: "Brown/tan bullseye rings\nCenter of the spot may fall out\nSpots can expand\nReduced fruit/flower production\nLeaf yellowing",
            fix: "Keep foliage dry\nRemove infected leaves\nApply fungicide"
        },
        {
            name: "Mosaic Virus",
            description: "A group of plant viruses that cause mottle patterns on leaves that resemble a mosaic. It affects vegetables, fruits, and ornamentals. They are transmitted by insects or contaminated tools.",
            appearance: "Irregular patterns of colors on leaves\nDistorted, stunted, or wrinkled leaves\nStunted flower/fruit growth\nColor breaks on fruit/flowers",
            fix: "None"
        },
        {
            name: "Yellow Curl Virus",
            description: "DNA viruses that affect plants like tomatoes, pepper, and beans (Solanaceae family). They are transmitted by whiteflies and cause yield loss due to stunted growth of fruit.",
            appearance: "Upward curling of leaves\nYellowing of leaves\nReduced branching\nStuted growth\nMalformatoin of flowers and fruit"
        }
    ];



    const filteredDiseases = diseases.filter(disease =>
        disease.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        disease.description.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className={`${cookies.darkMode ? "bg-gray-900 text-white" : "bg-gradient-to-r from-white via-gray-100 to-white text-black"} min-h-screen pb-20`}>
            <AIChatbot />
            {/* Header */}
            <div className="w-full py-20 px-4 mt-10 text-center">
                <motion.div
                    initial={{ y: -20, opacity: 0 }}
                    animate={{ y: 10, opacity: 1 }}
                    transition={{ duration: 0.5 }}
                >
                    <div className="flex justify-center items-center mb-2">
                        <h1 className={`text-2xl md:text-5xl font-bold bg-gradient-to-r bg-clip-text text-transparent ${cookies.darkMode ? "from-blue-600 to-sky-400" : "from-green-600 to-green-400"} `}>
                            Plant Disease Guide 🌱
                        </h1>
                    </div>
                    <p className="text-lg md:text-xl max-w-3xl mx-auto">
                        Comprehensive guide to identifying and treating common plant diseases
                    </p>
                </motion.div>
            </div>

            {/* Search Bar */}
            <div className="max-w-[90vw] mx-auto px-4 mb-8">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="flex flex-col md:flex-row gap-4 items-center"
                >
                    <div className="relative w-full">
                        <input
                            type="text"
                            placeholder="Search diseases or symptoms..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full p-4 pl-12 rounded-lg shadow-md text-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
                        />
                        <FontAwesomeIcon
                            icon={faSearch}
                            className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 text-xl"
                        />
                    </div>
                </motion.div>
            </div>

            {/* Disease Cards */}
            <div className="max-w-[90vw] mx-auto px-4">
                {filteredDiseases.length > 0 ? (
                    <motion.div
                        ref={containerRef}
                        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-4"
                    >
                        {filteredDiseases.map((disease, index) => (
                            <motion.div
                                key={index}
                                className={`rounded-xl overflow-hidden shadow-lg transition-all flex flex-col ${cookies.darkMode
                                    ? "bg-gradient-to-br from-sky-900 via-neutral-800 to-blue-900 border-white-600"
                                    : "bg-gradient-to-br from-green-200 via-neutral-200 to-emerald-200 border-lime-700"
                                    } border`}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.3, delay: index * 0.05 }}
                            >
                                <div className="p-6 flex-grow">
                                    <h3 className="font-bold text-lg mb-2">{disease.name}</h3>
                                    <p className={`text-sm line-clamp-3 mb-4 ${cookies.darkMode ? "text-white" : "text-gray-800"
                                        }`}>
                                        {disease.description}
                                    </p>
                                </div>
                                <button
                                    onClick={() => {
                                        setSelectedDisease(disease);
                                        setIsModalOpen(true);
                                    }}
                                    className={`flex items-center justify-center gap-2 p-3 border-t ${cookies.darkMode
                                        ? "border-gray-700 text-blue-400 hover:bg-cyan-700"
                                        : "border-neutral-300 text-green-600 hover:bg-gray-50"
                                        } transition-colors`}
                                >
                                    <span>Learn More</span>
                                    <FontAwesomeIcon icon={faChevronRight} />
                                </button>


                            </motion.div>
                        ))}
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-center py-20"
                    >
                        <FontAwesomeIcon
                            icon={faLeaf}
                            className="text-5xl text-gray-400 mb-4"
                        />
                        <p className="text-xl text-gray-500 mb-4">No diseases found matching your search</p>
                        <button
                            onClick={() => setSearchTerm('')}
                            className={`px-6 py-2 rounded-lg transition-colors ${cookies.darkMode
                                ? "bg-blue-600 hover:bg-blue-700 text-white"
                                : "bg-blue-500 hover:bg-blue-600 text-white"
                                }`}
                        >
                            Reset Search
                        </button>
                    </motion.div>
                )}
            </div>

            
            

        {/* Disease Modal */}
        {isModalOpen && selectedDisease && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
                ref={constraintsRef}
            >
                <motion.div
                    className="flex items-center justify-center min-h-screen p-4 text-center"
                    onClick={() => setIsModalOpen(false)}
                    drag
                    dragConstraints={constraintsRef}
                >
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.3, ease: "easeOut" }}
                        onClick={(e) => e.stopPropagation()}
                        className={`w-full max-w-2xl rounded-2xl shadow-2xl border ${
                            cookies.darkMode
                                ? "bg-gradient-to-br from-gray-900 via-sky-950 to-teal-900 border-gray-700"
                                : "bg-gradient-to-br from-sky-100 via-stone-200 to-blue-100 border-gray-300"
                        }`}
                    >
                        {/*Modal closing/opening*/}
                        <div className="flex items-start justify-between p-6 border-b border-opacity-20">
                            <h3 className={`text-3xl font-bold mx-auto ${cookies.darkMode ? "text-white" : "text-gray-900"}`}>
                                {selectedDisease.name}
                            </h3>
                        </div>

                        {/*Info Text*/}
                        <div className="px-6 py-6 space-y-6 text-center">
                            <p className={`text-lg leading-relaxed ${cookies.darkMode ? "text-gray-300" : "text-gray-700"}`}>
                                {selectedDisease.description}
                            </p>

                            {/*How to fix and how to identify stuff*/}
                            <div className="space-y-4">
                                {/*How to Identify - Gang help me out here*/}
                                <div className="rounded-xl overflow-hidden shadow-md">
                                    <button
                                        onClick={() => toggleSection('identify')}
                                        className={`w-full px-6 py-4 text-xl font-semibold flex items-center justify-between ${
                                            cookies.darkMode
                                                ? "bg-gray-700 text-white hover:bg-gray-600"
                                                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                                        }`}
                                    >
                                        How to Identify
                                        <FontAwesomeIcon icon={openSection === 'identify' ? faArrowDown : faArrowDown} rotation={openSection === 'identify' ? 180 : 0} />
                                    </button>
                                    <AnimatePresence>
                                        {openSection === 'identify' && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: "auto", opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                className={`px-6 py-4 ${
                                                    cookies.darkMode ? "bg-gray-800 text-indigo-300" : "bg-white text-gray-700"
                                                }`}
                                            >
                                                {selectedDisease.appearance.split("\n").map((line, idx) => (
                                                    <p key={idx} className="flex justify-center items-center gap-2 mb-2">
                                                        <span>•</span> <span>{line}</span>
                                                    </p>
                                                ))}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>

                                {/* How to Fix - Hopefully I can fix my sanity from how long this took*/}
                                <div className="rounded-xl overflow-hidden shadow-md">
                                    <button
                                        onClick={() => toggleSection('fix')}
                                        className={`w-full px-6 py-4 text-xl font-semibold flex items-center justify-between ${
                                            cookies.darkMode
                                                ? "bg-gray-700 text-white hover:bg-gray-600"
                                                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                                        }`}
                                    >
                                        How to Fix
                                        <FontAwesomeIcon icon={openSection === 'fix' ? faArrowDown : faArrowDown} rotation={openSection === 'fix' ? 180 : 0} />
                                    </button>
                                    <AnimatePresence>
                                        {openSection === 'fix' && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: "auto", opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                className={`px-6 py-4 ${
                                                    cookies.darkMode ? "bg-gray-800 text-indigo-300" : "bg-white text-gray-700"
                                                }`}
                                            >
                                                {selectedDisease.fix.split("\n").map((line, idx) => (
                                                    <p key={idx} className="flex justify-center items-center gap-2 mb-2">
                                                        <span>•</span> <span>{line}</span>
                                                    </p>
                                                ))}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>
                        </div>

                        {/*Stuff outside where the main text is*/}
                        <div className="px-6 py-4 border-t border-opacity-20 text-center">
                            <button
                                onClick={() => setIsModalOpen(false)}
                                className={`px-6 py-2 text-lg font-semibold rounded-xl transition ${
                                    cookies.darkMode
                                        ? "bg-sky-700 hover:bg-sky-600 text-white"
                                        : "bg-sky-400 hover:bg-sky-500 text-white"
                                }`}
                            >
                                Close
                            </button>
                        </div>
                    </motion.div>
                </motion.div>
            </div>
        )}


        </div>
    );
}
