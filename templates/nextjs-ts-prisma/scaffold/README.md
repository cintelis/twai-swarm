# Next.js + TypeScript + Prisma starter

Full-stack starter: Next.js 14 App Router, TypeScript, Prisma ORM (SQLite default), Vitest.

## Quick start

```bash
npm install
cp .env.example .env
npx prisma migrate dev --name init   # creates prisma/dev.db + applies schema
npm run dev                           # http://localhost:3000
```

## Layout

```
app/
  layout.tsx          # Root layout
  page.tsx            # Server Component landing page (queries DB)
  api/items/route.ts  # GET /api/items + POST /api/items
lib/
  prisma.ts           # PrismaClient singleton (Next.js dev hot-reload safe)
prisma/
  schema.prisma       # Item model — replace with your domain
tests/
  items.test.ts       # Vitest sample
```

## Verify

```bash
./verify.sh                  # full pipeline (lint + typecheck + smoke + test)
./verify.sh lint             # next lint
./verify.sh typecheck        # tsc --noEmit
./verify.sh smoke            # next build
./verify.sh test             # vitest run
```

## Swap to Postgres

1. `prisma/schema.prisma` → set `provider = "postgresql"`
2. `.env` → set `DATABASE_URL` to your Postgres URL
3. `npx prisma migrate dev`
