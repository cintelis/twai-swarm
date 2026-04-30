# spring-boot-jpa starter

Reusable starter template: **Spring Boot 3 + Java 21 + JPA + PostgreSQL + React/Vite + JWT auth + Testcontainers + Docker**.

## What's pre-wired

- **Backend:** Spring Boot 3, Spring Security with JWT (`/api/auth/login`), JPA/Hibernate, PostgreSQL.
- **Frontend:** Vite + React 19 + TypeScript, RTK Query, react-router, a `LoginPage` that hits `/api/auth/login` and a `HomePage` that hits `/api/hello` (public) and `/api/hello/me` (protected).
- **Tests:** JUnit 5 + Spring MockMvc for the API, Testcontainers for Postgres integration tests, Vitest + React Testing Library for the frontend.
- **Build / run:** multi-stage `Dockerfile` (frontend → backend → JRE-only runtime image), `docker-compose.yml` brings up Postgres + the app.

The default seeded user (in-memory store) is `testuser` / `SecurePass123!` — replace `InMemoryUserStore` with a real `UserDetailsService` before shipping.

## Quick start

```bash
# 1. Bring up Postgres + app
docker compose up --build

# 2. Smoke test
curl http://localhost:8080/api/hello
# {"status":"ok","message":"hello from the spring-boot-jpa template"}

# 3. Get a token, hit the protected endpoint
TOKEN=$(curl -s -X POST http://localhost:8080/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"testuser","password":"SecurePass123!"}' | jq -r .token)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/api/hello/me
```

## Local dev (no Docker)

```bash
# Backend
mvn spring-boot:run

# Frontend (separate shell)
cd frontend
npm install
npm run dev
```

## Customising

When you fork this template:

1. **Rename the Java package.** It's `com.example.app`. Replace with your own (`com.acme.myservice`, etc.) and update the directory layout + the `package` declarations + `pom.xml` `groupId`.
2. **Rename the Maven artifact.** `groupId=com.example`, `artifactId=app` in `pom.xml`.
3. **Replace `InMemoryUserStore`** with your real user source (database, OIDC, etc.).
4. **Change the JWT signing secret.** It's hardcoded in `JwtUtil` for the template — load it from env or secrets manager.
5. **Add your domain models** under `com.example.app.persistence` next to (or replacing) the placeholder package.
6. **Replace `HelloController` + `HomePage.tsx`** with your real entry points.

## Verification

The template ships with a `verify.sh` script with stages:

```bash
bash verify.sh lint        # mvn compile + tsc
bash verify.sh typecheck   # tsc -b on the frontend
bash verify.sh smoke       # boot the app, curl /api/hello
bash verify.sh test        # mvn test + vitest run
bash verify.sh all         # everything (default)
```

`verify.sh all` is the gate — exit 0 means the scaffold is healthy.

## Source of truth

This scaffold lives at <https://github.com/TotallyWildAi/templates/tree/main/spring-boot-jpa> and is mirrored into the twai-swarm repo at `templates/spring-boot-jpa/scaffold/` via `scripts/sync_template.sh`.
