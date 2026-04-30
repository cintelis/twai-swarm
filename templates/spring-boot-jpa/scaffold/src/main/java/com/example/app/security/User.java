package com.example.app.security;

/**
 * User record representing an authenticated user with username, encoded password, and role.
 */
public record User(
    String username,
    String password,
    String role
) {
}
