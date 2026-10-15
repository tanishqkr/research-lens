/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        workbench: "var(--background)",
        surface: {
          DEFAULT: "var(--surface-base)",
          subtle: "var(--surface-subtle)",
          hover: "var(--surface-hover)",
          active: "var(--surface-active)",
        },
        ink: {
          DEFAULT: "var(--foreground)",
          muted: "var(--foreground-muted)",
          subtle: "var(--foreground-subtle)",
        },
        border: {
          subtle: "var(--border-subtle)",
          DEFAULT: "var(--border-default)",
          strong: "var(--border-strong)",
        },
        accent: {
          DEFAULT: "var(--accent-primary)",
          hover: "var(--accent-primary-hover)",
          surface: "var(--accent-surface)",
          border: "var(--accent-border)",
          text: "var(--accent-text)",
        },
        evidence: {
          supports: {
            bg: "var(--evidence-supports-bg)",
            border: "var(--evidence-supports-border)",
            text: "var(--evidence-supports-text)",
          },
          contradicts: {
            bg: "var(--evidence-contradicts-bg)",
            border: "var(--evidence-contradicts-border)",
            text: "var(--evidence-contradicts-text)",
          },
          qualifies: {
            bg: "var(--evidence-qualifies-bg)",
            border: "var(--evidence-qualifies-border)",
            text: "var(--evidence-qualifies-text)",
          },
          incomparable: {
            bg: "var(--evidence-incomparable-bg)",
            border: "var(--evidence-incomparable-border)",
            text: "var(--evidence-incomparable-text)",
          },
          insufficient: {
            bg: "var(--evidence-insufficient-bg)",
            border: "var(--evidence-insufficient-border)",
            text: "var(--evidence-insufficient-text)",
          },
        },
        ring: "var(--ring-focus)",
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        full: "var(--radius-full)",
      },
      boxShadow: {
        card: "var(--shadow-card)",
        "card-hover": "var(--shadow-card-hover)",
        popover: "var(--shadow-popover)",
      },
      spacing: {
        4.5: "1.125rem",
        5.5: "1.375rem",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          '"Helvetica Neue"',
          "Arial",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          '"SF Mono"',
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
