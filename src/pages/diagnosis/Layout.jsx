import { Outlet } from "react-router";

export default function DiagnosisLayout() {
    return (
        <main className="w-screen h-screen">
            <Outlet />
        </main>
    );
}
