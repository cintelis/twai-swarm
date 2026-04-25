// Vitest setup: import jest-dom matchers (toBeInTheDocument, toHaveClass…)
// and run global cleanup between tests so renders don't leak.
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});
