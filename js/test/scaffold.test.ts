import { describe, expect, it } from "vitest";
import { HIVE_BOT_WEB_VERSION } from "../src/index.js";

describe("package scaffold", () => {
  it("builds and runs", () => {
    expect(HIVE_BOT_WEB_VERSION).toBe("0.1.0");
  });
});
