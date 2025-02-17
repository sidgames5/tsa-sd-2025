import { useState } from "react";
import { Link } from "react-router";

export default function UploadPage() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState("");

    const handleImageChange = (event) => {
        const file = event.target.files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
    };

    const handleUpload = async () => {
        if (!image) return alert("Please select an image.");

        const formData = new FormData();
        formData.append("image", image);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            setResult(data.message); // Display AI result
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    };

    return (
        <div className="p-4">
            <input type="file" accept="image/*" onChange={handleImageChange} />
            {preview && (
                <img src={preview} alt="Preview" className="w-40 mt-2" />
            )}
            <button
                onClick={handleUpload}
                className="bg-blue-500 text-white px-4 py-2 mt-2"
            >
                Upload & Analyze
            </button>
            {result && <p className="mt-2">Result: {result}</p>}
        </div>
    );
}
