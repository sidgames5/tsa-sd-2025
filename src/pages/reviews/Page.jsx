import React, { useState, useEffect, useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPerson } from "@fortawesome/free-solid-svg-icons";

export default function ReviewsPage() {
  const [reviews, setReviews] = useState([]);
  const [formData, setFormData] = useState({
    name: "",
    message: "",
    profileImage: null,
  });
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCameraAllowed, setIsCameraAllowed] = useState(false);
  const [videoStream, setVideoStream] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    fetch("/api/get-reviews")
      .then((res) => res.json())
      .then((data) => setReviews(data.reviews || []));
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const openModal = () => {
    setIsModalOpen(true);
    setIsCameraAllowed(false);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop());
      setVideoStream(null);
    }
  };

  const handleCameraPermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setIsCameraAllowed(true);
      setVideoStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert("Camera access denied or not available.");
    }
  };

  const handleCapture = () => {
    const canvas = document.createElement("canvas");
    const video = videoRef.current;
    if (!video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (blob) {
        setFormData((prev) => ({ ...prev, profileImage: blob }));
      }
    }, "image/jpeg");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("name", formData.name);
    form.append("message", formData.message);
    if (formData.profileImage) {
      form.append("profileImage", formData.profileImage);
    }

    const res = await fetch("/api/submit-review", {
      method: "POST",
      body: form,
    });

    const result = await res.json();
    if (result.success) {
      setReviews([result.review, ...reviews]);
      setFormData({ name: "", message: "", profileImage: null });
      closeModal();
    } else {
      alert(result.error || "Submission failed.");
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-16">
      <div className="mt-20"></div>

      <div className="text-center mb-12">
        <div className="text-5xl font-bold text-yellow-500">⭐⭐⭐⭐⭐</div>
        <h1 className="text-4xl font-extrabold text-gray-800 mt-2">4.8 STARS</h1>
        <p className="text-gray-600 text-lg">Based on 38,792 Reviews</p>
      </div>

      <h2 className="text-4xl font-bold text-center text-green-800 mb-10">Testimonials</h2>

      <div className="text-center mb-10">
        <button
          onClick={openModal}
          className="bg-green-600 text-white py-2 px-6 rounded-full text-lg hover:bg-green-700 transition-all"
        >
          Add a Review
        </button>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 flex justify-center items-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-6 rounded-xl shadow-lg w-full max-w-lg">
            <h3 className="text-2xl font-semibold mb-4 text-gray-800">Submit Your Review</h3>
            <form onSubmit={handleSubmit} className="space-y-4">
              <input
                type="text"
                name="name"
                placeholder="Your name (optional)"
                value={formData.name}
                onChange={handleChange}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 text-gray-800"
              />
              <textarea
                name="message"
                placeholder="Write your testimonial..."
                value={formData.message}
                onChange={handleChange}
                required
                className="w-full p-3 border border-gray-300 rounded-md h-28 resize-none focus:ring-2 focus:ring-green-500 text-gray-800"
              />

              <div className="flex items-center space-x-4">
                <button
                  type="button"
                  onClick={handleCameraPermission}
                  className="flex items-center space-x-2 bg-gray-200 py-2 px-4 rounded-full hover:bg-gray-300"
                >
                  <FontAwesomeIcon icon={faPerson} className="text-xl text-gray-700" />
                  <span className="text-gray-700">Take Profile Picture</span>
                </button>
              </div>

              {isCameraAllowed && (
                <div className="flex flex-col items-center space-y-4">
                  <video
                    ref={videoRef}
                    width="200"
                    height="200"
                    autoPlay
                    className="rounded-xl border"
                  />
                  <button
                    type="button"
                    onClick={handleCapture}
                    className="bg-green-500 text-white py-2 px-4 rounded-full hover:bg-green-600"
                  >
                    Capture Photo
                  </button>
                </div>
              )}

              <div className="flex justify-end space-x-4">
                <button
                  type="button"
                  onClick={closeModal}
                  className="bg-gray-300 py-2 px-4 rounded-md hover:bg-gray-400"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-green-600 text-white py-2 px-6 rounded-full hover:bg-green-700"
                >
                  Submit Review
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {reviews.length === 0 && (
          <p className="text-center text-gray-600 col-span-2">No reviews yet.</p>
        )}
        {reviews.map((rev, idx) => (
          <div
            key={idx}
            className="bg-gray-100 p-6 rounded-xl shadow flex flex-col items-center text-center"
          >
            {rev.image ? (
              <img
                src={rev.image}
                alt="Profile"
                className="w-16 h-16 rounded-full object-cover mb-4"
              />
            ) : (
              <div className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center mb-4">
                <FontAwesomeIcon icon={faPerson} className="text-3xl text-gray-600" />
              </div>
            )}
            <p className="italic text-gray-700">"{rev.message}"</p>
            <p className="mt-4 font-semibold text-gray-800">{rev.name || "Anonymous"}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
