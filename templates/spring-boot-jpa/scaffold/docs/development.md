# Development Guide

## Project Structure

```
calculator/
├── src/main/java/com/bench/
│   ├── CalculatorApp.java              # Spring Boot entry point, SecurityFilterChain config
│   ├── CalculatorController.java       # REST endpoint for /api/calculate
│   ├── Calculator.java                 # Main calculator logic dispatcher
│   ├── Adder.java                      # Addition operation
│   ├── Subtractor.java                 # Subtraction operation
│   ├── Multiplier.java                 # Multiplication operation
│   ├── Divider.java                    # Division operation
│   ├── security/
│   │   ├── AuthController.java         # REST endpoint for /api/auth/login
│   │   ├── JwtUtil.java                # JWT token generation and validation
│   │   ├── JwtAuthFilter.java          # Filter for token validation on requests
│   │   ├── LoginRequest.java           # DTO for login request
│   │   └── User.java                   # JPA entity for users table
│   └── persistence/
│       ├── CalculationService.java     # Service for recording calculations
│       ├── Calculation.java            # JPA entity for calculations table
│       └── CalculationRepository.java  # Spring Data JPA repository
├── src/test/java/com/bench/
│   ├── CalculatorTest.java             # Unit tests for Calculator
│   ├── AdderTest.java                  # Unit tests for Adder
│   ├── CalculatorControllerTest.java   # Integration tests with Testcontainers
│   └── AuthControllerTest.java         # Integration tests for login endpoint
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LoginForm.tsx           # Login form component
│   │   │   ├── CalculatorForm.tsx      # Calculator input form
│   │   │   └── HistoryList.tsx         # Display calculation history
│   │   ├── api/
│   │   │   └── calculatorApi.ts        # RTK Query API slice
│   │   ├── store/
│   │   │   └── store.ts                # Redux store configuration
│   │   ├── App.tsx                     # Main app component with routing
│   │   └── main.tsx                    # React entry point
│   ├── package.json                    # Frontend dependencies
│   ├── vite.config.ts                  # Vite build configuration
│   └── tsconfig.json                   # TypeScript configuration
├── pom.xml                             # Maven build configuration
├── Dockerfile                          # Multi-stage Docker build
├── docker-compose.yml                  # Local development with PostgreSQL
└── README.md                           # Project documentation
```

## Adding a New Operation

### Step 1: Create Operation Class

Create `src/main/java/com/bench/NewOperation.java`:
```java
public class NewOperation {
    public static int execute(int a, int b) {
        // Implement operation logic
        return result;
    }
}
```

### Step 2: Update Calculator.java

Add case to `Calculator.calculate()` method:
```java
case "newop" -> NewOperation.execute(a, b);
```

### Step 3: Add Unit Tests

Create `src/test/java/com/bench/NewOperationTest.java`:
```java
@Test
void testNewOperation() {
    assertEquals(expectedResult, NewOperation.execute(a, b));
}
```

### Step 4: Update Integration Tests

Add test case to `CalculatorControllerTest.java`:
```java
@Test
void testCalculateNewOp() throws Exception {
    mockMvc.perform(get("/api/calculate?a=10&b=5&op=newop")
        .header("Authorization", "Bearer " + token))
        .andExpect(status().isOk())
        .andExpect(jsonPath("$.result").value(expectedResult));
}
```

## Adding a New API Endpoint

### Step 1: Create Controller Method

Add method to `CalculatorController.java`:
```java
@GetMapping("/api/newendpoint")
public ResponseEntity<Map<String, Object>> newEndpoint(@RequestParam String param) {
    // Implement endpoint logic
    return ResponseEntity.ok(response);
}
```

### Step 2: Update Security Configuration

Modify `CalculatorApp.java` SecurityFilterChain if endpoint requires auth:
```java
.requestMatchers("/api/newendpoint").authenticated()  // or .permitAll()
```

### Step 3: Add Integration Tests

Create test in `CalculatorControllerTest.java`:
```java
@Test
void testNewEndpoint() throws Exception {
    mockMvc.perform(get("/api/newendpoint?param=value")
        .header("Authorization", "Bearer " + token))
        .andExpect(status().isOk());
}
```

### Step 4: Update Frontend (if needed)

Add RTK Query endpoint to `frontend/src/api/calculatorApi.ts`:
```typescript
newEndpoint: builder.query({
    query: (param) => `/api/newendpoint?param=${param}`,
}),
```

## Running Tests Locally

**All tests:**
```bash
mvn clean test && cd frontend && npm test
```

**Backend only:**
```bash
mvn test
```

**Specific test class:**
```bash
mvn test -Dtest=CalculatorControllerTest
```

**Frontend only:**
```bash
cd frontend && npm test
```

## Local Development with Docker Compose

```bash
docker-compose up
```

Starts PostgreSQL on port 5432. Backend connects automatically via `spring.datasource.url=jdbc:postgresql://postgres:5432/calculator`.
