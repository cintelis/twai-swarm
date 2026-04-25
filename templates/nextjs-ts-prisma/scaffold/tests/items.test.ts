// Pure-logic test that doesn't touch Prisma — keeps the verify pipeline
// fast (no DB needed) while proving the test runner is wired. Replace with
// real route-handler integration tests once you've added domain logic.
import { describe, expect, it } from "vitest";

function isValidName(value: unknown): value is string {
  return typeof value === "string" && value.trim() !== "";
}

describe("isValidName", () => {
  it("accepts a non-empty string", () => {
    expect(isValidName("widget")).toBe(true);
  });

  it("rejects empty / whitespace / non-string values", () => {
    expect(isValidName("")).toBe(false);
    expect(isValidName("   ")).toBe(false);
    expect(isValidName(undefined)).toBe(false);
    expect(isValidName(42)).toBe(false);
  });
});
