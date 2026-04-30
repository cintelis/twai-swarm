package com.example.app.security;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * In-memory user store that pre-populates a test user with BCrypt-encoded password.
 * Used by CustomUserDetailsService to load user credentials.
 */
@Component
public class InMemoryUserStore {

    private final Map<String, User> users = new HashMap<>();

    public InMemoryUserStore() {
        BCryptPasswordEncoder encoder = new BCryptPasswordEncoder();
        String encodedPassword = encoder.encode("SecurePass123!");
        users.put("testuser", new User("testuser", encodedPassword, "USER"));
    }

    /**
     * Find a user by username.
     *
     * @param username the username to search for
     * @return Optional containing the User if found, empty otherwise
     */
    public Optional<User> findByUsername(String username) {
        return Optional.ofNullable(users.get(username));
    }
}
