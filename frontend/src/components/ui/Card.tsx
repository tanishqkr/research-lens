import React from "react";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "base" | "subtle" | "hover";
  children?: React.ReactNode;
}

export function Card({
  children,
  variant = "base",
  className = "",
  ...props
}: CardProps) {
  const bgClass =
    variant === "subtle"
      ? "bg-surface-subtle"
      : variant === "hover"
      ? "bg-surface hover:bg-surface-hover hover:border-border-strong transition-colors"
      : "bg-surface";

  return (
    <div
      className={`rounded-lg border border-border ${bgClass} shadow-card ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  children,
  className = "",
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`p-4 sm:p-5 border-b border-border ${className}`}>
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  className = "",
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <h3 className={`text-base font-semibold text-ink flex items-center gap-2 ${className}`}>
      {children}
    </h3>
  );
}

export function CardDescription({
  children,
  className = "",
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return <p className={`text-sm text-ink-muted mt-1 ${className}`}>{children}</p>;
}

export function CardContent({
  children,
  className = "",
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return <div className={`p-4 sm:p-5 ${className}`}>{children}</div>;
}

export function CardFooter({
  children,
  className = "",
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`p-4 sm:p-5 border-t border-border bg-surface-subtle/50 rounded-b-lg ${className}`}>
      {children}
    </div>
  );
}
