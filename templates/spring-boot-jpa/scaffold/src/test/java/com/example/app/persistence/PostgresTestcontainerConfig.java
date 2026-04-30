package com.example.app.persistence;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;

import java.util.concurrent.atomic.AtomicBoolean;

@TestConfiguration
public class PostgresTestcontainerConfig {

    private static final PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:16-alpine")
            .withDatabaseName("app_test")
            .withUsername("test")
            .withPassword("test");

    private static final AtomicBoolean containerStarted = new AtomicBoolean(false);

    static {
        try {
            postgres.start();
            containerStarted.set(true);
        } catch (Exception e) {
            System.err.println("Warning: Could not start PostgreSQL container: " + e.getMessage());
            containerStarted.set(false);
        }
    }

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        if (containerStarted.get()) {
            registry.add("spring.datasource.url", postgres::getJdbcUrl);
            registry.add("spring.datasource.username", postgres::getUsername);
            registry.add("spring.datasource.password", postgres::getPassword);
        }
    }

    public static boolean isContainerRunning() {
        return containerStarted.get();
    }
}
