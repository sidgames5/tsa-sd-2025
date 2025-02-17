import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/global.css';
import { BrowserRouter, Route, Routes } from 'react-router';
import HomeLayout from './pages/Layout';
import HomePage from './pages/Page';
import UploadLayout from './pages/upload/Layout';
import UploadPage from './pages/upload/Page';
import DiagnosisLayout from './pages/diagnosis/Layout';
import DiagnosisPage from './pages/diagnosis/Page';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <BrowserRouter>
        <Routes>
            <Route path="/" element={<HomeLayout />}>
                <Route index element={<HomePage />} />
                <Route path="upload" element={<UploadLayout />}>
                    <Route index element={<UploadPage />} />
                </Route>
                <Route path="diagnosis" element={<DiagnosisLayout />}>
                    <Route index element={<DiagnosisPage />} />
                </Route>
            </Route>
        </Routes>
    </BrowserRouter>
);
