package com.example.app.security;

/**
 * LoginRequest record for JSON deserialization of login endpoint request body.
 * Contains username and password fields.
 */
public record LoginRequest(
    String username,
    String password
) {
}

