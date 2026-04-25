import { describe, expect, it } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("renders the heading", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /vite \+ react \+ tailwind/i })).toBeInTheDocument();
  });

  it("increments the counter on click", () => {
    render(<App />);
    const btn = screen.getByRole("button", { name: /count: 0/i });
    fireEvent.click(btn);
    expect(screen.getByRole("button", { name: /count: 1/i })).toBeInTheDocument();
  });
});
