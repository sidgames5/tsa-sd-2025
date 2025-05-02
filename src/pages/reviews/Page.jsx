import React, { useState, useEffect } from "react";
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

  // Fetch previous reviews when component mounts
  useEffect(() => {
    fetch("api/get-reviews")
      .then((res) => res.json())
      .then((data) => setReviews(data.reviews || []));
  }, []);

  // Handle input changes (text, file)
  const handleChange = (e) => {
    const { name, value, files } = e.target;
    if (name === "profileImage") {
      setFormData((prev) => ({ ...prev, profileImage: files[0] }));
    } else {
      setFormData((prev) => ({ ...prev, [name]: value }));
    }
  };

  // Open the modal for adding a new review
  const openModal = () => {
    setIsModalOpen(true);
    setIsCameraAllowed(false); // Reset camera choice
  };

  // Close the modal
  const closeModal = () => {
    setIsModalOpen(false);
  };

  // Handle camera access
  const handleCameraPermission = () => {
    setIsCameraAllowed(true);
  };

  // Submit the review
  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("name", formData.name);
    form.append("message", formData.message);
    if (formData.profileImage) {
      form.append("profileImage", formData.profileImage);
    }

    const res = await fetch("/submit-review", {
      method: "POST",
      body: form,
    });

    const result = await res.json();
    if (result.success) {
      setReviews([result.review, ...reviews]);
      setFormData({ name: "", message: "", profileImage: null });
      closeModal(); // Close the modal after submission
    } else {
      alert(result.error || "Submission failed.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 bg-gray-100">
      <h2 className="text-4xl font-bold text-center mb-8 text-green-700">Testimonials</h2>

      <div className="text-center mb-8">
        <button
          onClick={openModal}
          className="bg-green-600 text-white py-2 px-6 rounded-full text-lg hover:bg-green-700 transition-all"
        >
          Add a Review
        </button>
      </div>

      {/* Modal for Review Submission */}
      {isModalOpen && (
        <div className="fixed inset-0 flex justify-center items-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-lg">
            <h3 className="text-2xl font-semibold mb-4 text-gray-800">Submit Your Review</h3>

            <form onSubmit={handleSubmit} className="space-y-4">
              <input
                type="text"
                name="name"
                placeholder="Your name (optional)"
                value={formData.name}
                onChange={handleChange}
                className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 text-gray-800"
              />
              <textarea
                name="message"
                placeholder="Write your testimonial..."
                value={formData.message}
                onChange={handleChange}
                required
                className="w-full p-3 border border-gray-300 rounded-md h-28 resize-none focus:outline-none focus:ring-2 focus:ring-green-500 text-gray-800"
              />

              {/* Profile Image - Camera Button */}
              <div className="flex items-center space-x-4">
                <button
                  type="button"
                  onClick={handleCameraPermission}
                  className="flex items-center space-x-2 bg-gray-200 py-2 px-4 rounded-full hover:bg-gray-300 transition-all"
                >
                  <FontAwesomeIcon icon={faPerson} className="text-xl text-gray-700" />
                  <span className="text-gray-700">Take Profile Picture</span>
                </button>

                {isCameraAllowed && (
                  <video id="video" width="200" height="200" autoPlay />
                )}
              </div>

              <input
                type="file"
                name="profileImage"
                accept="image/*"
                onChange={handleChange}
                className="w-full"
              />

              <div className="flex justify-end space-x-4">
                <button
                  type="button"
                  onClick={closeModal}
                  className="bg-gray-300 py-2 px-4 rounded-md hover:bg-gray-400 transition-all"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-green-600 text-white py-2 px-6 rounded-full hover:bg-green-700 transition-all"
                >
                  Submit Review
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Reviews List */}
      <div className="mt-8">
        <h3 className="text-xl font-semibold mb-3 text-gray-800">What others are saying:</h3>
        {reviews.length === 0 && <p className="text-gray-600">No reviews yet.</p>}
        <div className="space-y-4">
          {reviews.map((rev, idx) => (
            <div
              key={idx}
              className="bg-white border p-4 rounded-lg shadow flex items-start space-x-4"
            >
              {rev.image ? (
                <img
                  src={rev.image}
                  alt="Profile"
                  className="w-12 h-12 rounded-full object-cover"
                />
              ) : (
                <FontAwesomeIcon
                  icon={faPerson}
                  className="text-4xl text-gray-400"
                />
              )}
              <div>
                <p className="font-semibold text-gray-800">{rev.name || "Anonymous"}</p>
                <p className="text-gray-700">{rev.message}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
