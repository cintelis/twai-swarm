import { useState } from "react";

export default function App() {
  const [count, setCount] = useState(0);
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6 bg-slate-50 text-slate-900">
      <h1 className="text-4xl font-bold">Vite + React + Tailwind</h1>
      <p className="text-slate-600">Replace this scaffold with your real UI.</p>
      <button
        type="button"
        onClick={() => setCount((c) => c + 1)}
        className="px-4 py-2 rounded bg-slate-900 text-white hover:bg-slate-700"
      >
        count: {count}
      </button>
    </main>
  );
}
