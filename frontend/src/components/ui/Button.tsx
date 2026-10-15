"use client";

import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";

interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  children?: React.ReactNode;
  variant?: "primary" | "secondary" | "outline" | "ghost";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
}

export function Button({
  children,
  variant = "primary",
  size = "md",
  isLoading = false,
  className = "",
  disabled,
  whileHover,
  whileTap,
  ...props
}: ButtonProps) {
  const sizeClasses = {
    sm: "px-3 py-1.5 text-xs font-medium",
    md: "px-4 py-2 text-sm font-medium",
    lg: "px-5 py-2.5 text-base font-medium",
  }[size];

  const variantClasses = {
    primary:
      "bg-accent text-white hover:bg-accent-hover shadow-sm",
    secondary:
      "bg-surface-subtle text-ink hover:bg-surface-hover border border-border",
    outline:
      "bg-transparent text-ink hover:bg-surface-subtle border border-border",
    ghost:
      "bg-transparent text-ink-muted hover:text-ink hover:bg-surface-subtle",
  }[variant];

  const isDisabled = disabled || isLoading;

  return (
    <motion.button
      disabled={isDisabled}
      whileHover={isDisabled ? undefined : whileHover ?? { y: -2, transition: { duration: 0.15 } }}
      whileTap={isDisabled ? undefined : whileTap ?? { scale: 0.98 }}
      className={`inline-flex items-center justify-center gap-2 rounded-md transition-colors duration-150 focus-visible:outline-none disabled:opacity-50 disabled:pointer-events-none cursor-pointer ${sizeClasses} ${variantClasses} ${className}`}
      {...props}
    >
      {isLoading && (
        <svg
          className="animate-spin -ml-1 mr-2 h-4 w-4 text-current"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      )}
      {children}
    </motion.button>
  );
}
