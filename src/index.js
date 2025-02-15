import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/global.css';
import { BrowserRouter, Route, Routes } from 'react-router';
import HomeLayout from './pages/Layout';
import HomePage from './pages/Page';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <BrowserRouter>
        <Routes>
            <Route path="/" element={<HomeLayout />}>
                <Route index element={<HomePage />} />
            </Route>
        </Routes>
    </BrowserRouter>
);
