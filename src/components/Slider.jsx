import React, { useState } from 'react';
import { motion } from 'motion/react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowLeft, faArrowRight } from '@fortawesome/free-solid-svg-icons';

export default function ImageSlider() {
    const [currentIndex, setCurrentIndex] = useState(0);

    const images = [
        "480-360-max.png",
        "612-344-max.png",
        "1280-853-max.png"
    ];

    function imageLeft() {
        setCurrentIndex(currentIndex - 1);
        if (currentIndex === 0) {
            setCurrentIndex(images.length - 1);
        }
    }

    function imageRight() {
        setCurrentIndex(currentIndex + 1);
        if (currentIndex === images.length - 1) {
            setCurrentIndex(0);
        }
    }

    return (
        <div className="flex flex-col w-1/2 h-1/2">
            <div className="flex flex-row align-middle justify-center items-center gap-8 *:text-5xl">
                <motion.div className="cursor-pointer h-full" onClick={imageLeft}><FontAwesomeIcon icon={faArrowLeft} /></motion.div>
                <motion.img src={`assets/${images[currentIndex]}`} alt="" className="max-w-96 rounded-xl" />
                <motion.div className="cursor-pointer h-full" onClick={imageRight}><FontAwesomeIcon icon={faArrowRight} /></motion.div>
            </div>
        </div>
    );
}