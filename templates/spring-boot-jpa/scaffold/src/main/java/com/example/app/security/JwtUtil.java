package com.example.app.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Component;
import javax.crypto.SecretKey;
import java.util.Date;

/**
 * JWT utility for generating and validating JSON Web Tokens.
 * Uses HS256 signing with a 256-bit secret key.
 */
@Component
public class JwtUtil {

    private static final String SECRET_KEY = "MyVerySecureSecretKeyForJWTSigningWith256Bits!";
    private static final long EXPIRATION_TIME = 3600000;

    private final SecretKey key;

    public JwtUtil() {
        this.key = Keys.hmacShaKeyFor(SECRET_KEY.getBytes());
    }

    /**
     * Generate a JWT token for the given username.
     *
     * @param username the username to encode in the token
     * @return JWT token string signed with HS256
     */
    public String generateToken(String username) {
        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + EXPIRATION_TIME);

        return Jwts.builder()
            .subject(username)
            .issuedAt(now)
            .expiration(expiryDate)
            .signWith(key, SignatureAlgorithm.HS256)
            .compact();
    }

    /**
     * Extract the username (subject) from a JWT token.
     *
     * @param token the JWT token string
     * @return the username encoded in the token
     */
    public String extractUsername(String token) {
        Claims claims = Jwts.parser()
            .verifyWith(key)
            .build()
            .parseSignedClaims(token)
            .getPayload();
        return claims.getSubject();
    }

    /**
     * Validate a JWT token: check signature, expiration, and username match.
     *
     * @param token the JWT token string
     * @param username the expected username
     * @return true if token is valid and username matches, false otherwise
     */
    public boolean isTokenValid(String token, String username) {
        try {
            String extractedUsername = extractUsername(token);
            return extractedUsername.equals(username);
        } catch (Exception e) {
            return false;
        }
    }
}
