import * as motion from "motion/react-client";
import React from "react";
import "../components/CustomSlider"
import ImageSlider from "../components/Slider.jsx";

const images = [
    {
        imgURL:
            "https://images.pexels.com/photos/461198/pexels-photo-461198.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-1"
    },
    {
        imgURL:
            "https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-2"
    },
    {
        imgURL:
            "https://images.pexels.com/photos/1128678/pexels-photo-1128678.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-3"
    },
    {
        imgURL:
            "https://images.pexels.com/photos/54455/cook-food-kitchen-eat-54455.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260",
        imgAlt: "img-4"
    }
];

export default function App() {
    return (
        <div className="App flex flex-col justify-center items-center">
            {/* <CustomSlider>
                {images.map((image, index) => {
                    return <img key={index} src={image.imgURL} alt={image.imgAlt} />;
                })}
            </CustomSlider> */}
            <ImageSlider />
            <div className="info flex flex-col justify-center items-center">
                <h1>ReactJS Slider</h1>
                <h2>Features</h2>
                <ul>
                    <li>
                        <p>Autoplay</p>
                    </li>
                    <li>
                        <p>Next and Previous Buttons</p>
                    </li>
                    <li>
                        <p>Select a desired slide</p>
                    </li>
                </ul>
            </div>
        </div>
    );
}
