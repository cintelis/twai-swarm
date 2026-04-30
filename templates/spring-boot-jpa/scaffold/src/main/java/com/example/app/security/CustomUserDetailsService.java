package com.example.app.security;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;
import java.util.Collections;

/**
 * Custom UserDetailsService that loads user credentials from InMemoryUserStore.
 * Integrates with Spring Security's authentication framework.
 */
@Service
public class CustomUserDetailsService implements UserDetailsService {

    @Autowired
    private InMemoryUserStore userStore;

    /**
     * Load user by username from the in-memory store.
     *
     * @param username the username to load
     * @return UserDetails with username, encoded password, and ROLE_USER authority
     * @throws UsernameNotFoundException if user not found
     */
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        com.example.app.security.User user = userStore.findByUsername(username)
            .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));

        return new User(
            user.username(),
            user.password(),
            Collections.singletonList(new SimpleGrantedAuthority("ROLE_" + user.role()))
        );
    }
}
