'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router';
import { useCookies } from 'react-cookie';




export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState('');
  const [cookies, setCookie] = useCookies(['user', 'darkMode']);
  const navigate = useNavigate();

  const darkMode = cookies.darkMode === 'true';

  useEffect(() => {
    document.body.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const handleToggleMode = () => {
    setIsLogin(!isLogin);
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setMessage('');
  };

  const handlePasswordVisibility = () => {
    setShowPassword((prev) => !prev);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!email || !password || (!isLogin && !confirmPassword)) {
      setMessage('Please fill all fields.');
      return;
    }

    if (!isLogin) {
      if (password !== confirmPassword) {
        setMessage('Passwords do not match!');
        return;
      }

      const existingUser = localStorage.getItem(email);
      if (existingUser) {
        setMessage('User already exists. Please log in.');
        return;
      }

      localStorage.setItem(email, JSON.stringify({ password }));
      setMessage('Account created! Please log in.');
      setIsLogin(true);
    } else {
      const storedUser = localStorage.getItem(email);
      if (!storedUser) {
        setMessage('User not found. Please sign up.');
        return;
      }

      const { password: storedPassword } = JSON.parse(storedUser);
      if (password === storedPassword) {
        setMessage('Login successful!');
        setCookie('user', email, { path: '/', maxAge: 60 * 60 * 24 });
        navigate('/support');
      } else {
        setMessage('Incorrect password.');
      }
    }
  };

  return (
    <>
    
      <div
        className={`${
          darkMode
            ? 'bg-gray-900 text-white'
            : 'bg-gradient-to-br from-blue-100 via-white to-purple-100'
        } min-h-screen flex items-center justify-center px-4`}
      >
        <motion.div
          initial={{ y: 50, opacity: 0, scale: 0.95 }}
          animate={{ y: 0, opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`w-full max-w-md p-8 rounded-2xl shadow-xl border ${
            darkMode
              ? 'bg-gray-800 border-gray-700'
              : 'bg-white/80 border-gray-200 backdrop-blur-lg'
          }`}
        >
          <h2 className="text-3xl font-bold text-center mb-4">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>

          <AnimatePresence>
            {message && (
              <motion.p
                key="message"
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="text-center text-sm text-red-500 mb-4"
              >
                {message}
              </motion.p>
            )}
          </AnimatePresence>

          <form className="space-y-5" onSubmit={handleSubmit}>
            <div>
              <label htmlFor="email" className="block text-sm font-medium">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="you@example.com"
                className="mt-1 w-full px-4 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 dark:bg-gray-700 dark:border-gray-600"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  placeholder="Your password"
                  className="mt-1 w-full px-4 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 dark:bg-gray-700 dark:border-gray-600"
                />
                <button
                  type="button"
                  onClick={handlePasswordVisibility}
                  className="absolute right-3 top-2/3 transform -translate-y-1/2 text-sm text-blue-500 hover:underline"
                >
                  {showPassword ? 'Hide' : 'Show'}
                </button>
              </div>
            </div>

            <AnimatePresence>
              {!isLogin && (
                <motion.div
                  key="confirmPassword"
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                >
                  <label
                    htmlFor="confirmPassword"
                    className="block text-sm font-medium"
                  >
                    Confirm Password
                  </label>
                  <input
                    id="confirmPassword"
                    type={showPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                    placeholder="Re-enter password"
                    className="mt-1 w-full px-4 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 dark:bg-gray-700 dark:border-gray-600"
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="submit"
              className="w-full py-2 px-4 bg-blue-500 text-white font-semibold rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400 transition duration-200"
            >
              {isLogin ? 'Log In' : 'Sign Up'}
            </button>
          </form>

          <div className="mt-5 text-center text-sm text-gray-600 dark:text-gray-300">
            {isLogin ? (
              <>
                Don’t have an account?{' '}
                <button
                  onClick={handleToggleMode}
                  className="text-blue-500 hover:underline font-medium"
                >
                  Sign Up
                </button>
              </>
            ) : (
              <>
                Already have an account?{' '}
                <button
                  onClick={handleToggleMode}
                  className="text-blue-500 hover:underline font-medium"
                >
                  Log In
                </button>
              </>
            )}
          </div>
        </motion.div>
      </div>
    </>
  );
}
