import React, { useState } from 'react';
import { motion } from 'motion/react';

export default function ImageSlider() {
    const [currentIndex, setCurrentIndex] = useState(0);

    const images = [
        "480-360-max.png",
        "612-344-max.png",
        "612-344-max.png",
        "612-344-max.png",
    ];

    return (
        <div className="flex flex-row w-1/2 gap-4 *:rounded-xl *:h-64 overflow-y-visible overflow-x-scroll">
            {images.map((image) => <img src={`assets/${image}`} alt={image} />)}
        </div>
    );
}