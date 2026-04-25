# Vite + React + TypeScript + Tailwind starter

Frontend SPA starter: Vite 5, React 18, TypeScript strict, Tailwind v3, Vitest.

## Quick start

```bash
npm install
npm run dev          # http://localhost:5173
```

## Layout

```
src/
  main.tsx           # App entry — mounts <App /> on #root
  App.tsx            # Root component (replace with your real UI)
  App.test.tsx       # Vitest + Testing Library example
  index.css          # Tailwind directives — must come first
  test-setup.ts      # jest-dom matchers + per-test cleanup
index.html           # Vite HTML entry
tailwind.config.js   # Theme extension goes here
postcss.config.js    # Required by Tailwind
vite.config.ts       # Vite + Vitest config
tsconfig.json        # Strict + path alias @/ → src/
```

## Verify

```bash
./verify.sh                  # full pipeline (lint + typecheck + smoke + test)
./verify.sh lint             # eslint .
./verify.sh typecheck        # tsc --noEmit
./verify.sh smoke            # vite build
./verify.sh test             # vitest run
```

## Talking to a backend

This template is frontend-only. For the backend, point a separate API
service (FastAPI, Next.js Route Handlers, etc.) at the same domain or use
CORS. Set `VITE_API_BASE` in `.env` and read it via `import.meta.env`.
