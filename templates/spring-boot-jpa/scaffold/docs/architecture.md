# Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Browser                          │
│                   (React 19 + Vite)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LoginForm → CalculatorForm → HistoryList           │  │
│  │  (RTK Query for API calls)                          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP/HTTPS
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              Spring Boot REST API (Java 21)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AuthController (/api/auth/login)                   │  │
│  │  ├─ AuthenticationManager (BCrypt password check)    │  │
│  │  └─ JwtUtil (token generation)                      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  CalculatorController (/api/calculate)              │  │
│  │  ├─ JwtAuthFilter (token validation)                │  │
│  │  ├─ Calculator (business logic)                     │  │
│  │  └─ CalculationService (persistence)               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SecurityFilterChain (stateless JWT auth)           │  │
│  │  ├─ CSRF disabled                                   │  │
│  │  ├─ SessionCreationPolicy.STATELESS                 │  │
│  │  └─ /api/auth/login permitAll, /api/calculate auth │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         │ JDBC
                         │
┌────────────────────────▼─────────────────────────────────────┐
│           PostgreSQL 16 Database                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  users table (id, username, password_hash)          │  │
│  │  calculations table (id, a, b, op, result, user_id) │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Security Model

### Authentication Flow

1. **Login Request:** Client sends username + password to `/api/auth/login`
2. **Password Verification:** AuthenticationManager validates credentials against BCrypt-hashed password in database
3. **Token Generation:** JwtUtil generates a signed JWT token (HS256 algorithm)
4. **Token Storage:** Client stores token in localStorage or sessionStorage
5. **Authenticated Requests:** Client includes token in `Authorization: Bearer <token>` header
6. **Token Validation:** JwtAuthFilter intercepts requests, validates token signature and expiration
7. **Access Grant/Deny:** SecurityFilterChain permits `/api/calculate` only with valid token

### Key Security Features

- **Stateless Sessions:** No server-side session storage; all auth info in JWT
- **BCrypt Password Hashing:** Passwords never stored in plaintext; BCryptPasswordEncoder with strength 10
- **CSRF Protection:** Disabled for stateless API (not needed with JWT)
- **JWT Signing:** HS256 algorithm with secret key; token includes username and expiration
- **Token Expiration:** Tokens expire after configured duration (default 1 hour)
- **Filter Chain:** JwtAuthFilter runs before UsernamePasswordAuthenticationFilter

## Data Flow

### Calculation Request Flow

```
1. User logs in → POST /api/auth/login
   └─ Returns JWT token

2. User performs calculation → GET /api/calculate?a=10&b=5&op=add
   ├─ Request includes Authorization: Bearer <token>
   ├─ JwtAuthFilter validates token
   ├─ CalculatorController receives request
   ├─ Calculator.calculate(10, 5, 'add') → 15
   ├─ CalculationService.recordCalculation(10, 5, 'add', 15, username)
   │  └─ Inserts row into calculations table
   └─ Returns {a: 10, b: 5, op: 'add', result: 15}

3. User views history → Frontend queries RTK Query cache
   └─ Displays all calculations for authenticated user
```

## Deployment Architecture

### Docker Multi-Stage Build

**Stage 1 (Frontend Build):**
- Base: Node 20
- Build React + Vite application
- Output: Static files in `dist/`

**Stage 2 (Backend Build):**
- Base: Maven 3.9
- Compile Java source, run tests, package JAR
- Output: `target/adder-1.0.0.jar`

**Stage 3 (Runtime):**
- Base: JRE 21
- Copy JAR from stage 2
- Copy static files from stage 1 to `src/main/resources/static/`
- Expose port 8080
- Run: `java -jar adder-1.0.0.jar`

### Environment Variables

- `SPRING_DATASOURCE_URL`: PostgreSQL connection string (default: `jdbc:postgresql://localhost:5432/calculator`)
- `SPRING_DATASOURCE_USERNAME`: Database user (default: `postgres`)
- `SPRING_DATASOURCE_PASSWORD`: Database password
- `JWT_SECRET`: Secret key for signing JWT tokens
- `JWT_EXPIRATION`: Token expiration time in milliseconds (default: 3600000 = 1 hour)
