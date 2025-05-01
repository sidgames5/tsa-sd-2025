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
import FeaturesLayout from './pages/features/Layout';
import FeaturesPage from './pages/features/Page';
import ResultsLayout from './pages/results/Layout';
import ResultsPage from './pages/results/Page';
import LoginPage from './pages/login/Page';
import LoginLayout from './pages/login/Layout';
import SupportLayout from './support/Layout';
import { CookiesProvider } from 'react-cookie';
import "@fontsource/inter";
import "@fontsource/inter/400.css";
import "@fontsource/inter/400-italic.css";

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <CookiesProvider>
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<HomeLayout />}>
                    <Route index element={<HomePage />} />

                    <Route path="/upload" element={<UploadLayout />}>
                        <Route index element={<UploadPage />} />
                    </Route>

                    <Route path="/diagnosis" element={<DiagnosisLayout />}>
                        <Route index element={<DiagnosisPage />} />
                    </Route>

                    <Route path="/features" element={<FeaturesLayout />}>
                        <Route index element={<FeaturesPage />} />
                    </Route>

                    <Route path="/results" element={<ResultsLayout />}>
                        <Route index element={<ResultsPage />} />
                    </Route>

                    <Route path="/login" element={<LoginLayout />}>
                        <Route index element={<LoginPage />} />
                    </Route>
                    <Route path="/support" element={<SupportLayout />}>
                        <Route index element={<SupportPage />} />
                    </Route>
                </Route>
            </Routes>
        </BrowserRouter>
    </CookiesProvider>
);
