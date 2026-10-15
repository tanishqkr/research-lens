import React from "react";
import { DirectionLabel, RelationLabel, StudyDesignType, SubclaimStatus } from "@/types/scientific";

interface BadgeProps {
  children?: React.ReactNode;
  variant?:
    | "default"
    | "supports"
    | "contradicts"
    | "qualifies"
    | "incomparable"
    | "insufficient"
    | "accent"
    | "outline";
  size?: "sm" | "md";
  className?: string;
}

export function Badge({
  children,
  variant = "default",
  size = "md",
  className = "",
}: BadgeProps) {
  const sizeClasses = size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs font-semibold";

  const variantClasses = {
    default: "bg-surface-subtle text-ink-muted border border-border",
    supports: "bg-evidence-supports-bg text-evidence-supports-text border border-evidence-supports-border",
    contradicts: "bg-evidence-contradicts-bg text-evidence-contradicts-text border border-evidence-contradicts-border",
    qualifies: "bg-evidence-qualifies-bg text-evidence-qualifies-text border border-evidence-qualifies-border",
    incomparable: "bg-evidence-incomparable-bg text-evidence-incomparable-text border border-evidence-incomparable-border",
    insufficient: "bg-evidence-insufficient-bg text-evidence-insufficient-text border border-evidence-insufficient-border",
    accent: "bg-accent-surface text-accent-text border border-accent-border",
    outline: "bg-transparent text-ink border border-border",
  }[variant];

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-sm tracking-wide ${sizeClasses} ${variantClasses} ${className}`}
    >
      {children}
    </span>
  );
}

export function RelationBadge({ relation }: { relation: RelationLabel }) {
  const mapping: Record<RelationLabel, { label: string; variant: BadgeProps["variant"] }> = {
    SUPPORTS_OR_CONSISTENT: { label: "SUPPORTS", variant: "supports" },
    CONTRADICTS: { label: "CONTRADICTS", variant: "contradicts" },
    QUALIFIES_OR_PARTIAL_DISAGREEMENT: { label: "QUALIFIES", variant: "qualifies" },
    INCOMPARABLE_CONDITIONS: { label: "INCOMPARABLE", variant: "incomparable" },
    INSUFFICIENT_EVIDENCE: { label: "INSUFFICIENT EVIDENCE", variant: "insufficient" },
  };

  const item = mapping[relation] || { label: relation, variant: "default" };
  return <Badge variant={item.variant}>{item.label}</Badge>;
}

export function DirectionBadge({ direction }: { direction: DirectionLabel }) {
  const mapping: Record<DirectionLabel, { label: string; variant: BadgeProps["variant"] }> = {
    POSITIVE: { label: "POSITIVE (+)", variant: "supports" },
    NEGATIVE: { label: "NEGATIVE (−)", variant: "contradicts" },
    NO_DIFFERENCE: { label: "NO DIFFERENCE", variant: "incomparable" },
    MIXED: { label: "MIXED EFFECTS", variant: "qualifies" },
    UNCLEAR: { label: "UNCLEAR", variant: "insufficient" },
  };

  const item = mapping[direction] || { label: direction, variant: "default" };
  return <Badge variant={item.variant} size="sm">{item.label}</Badge>;
}

export function StudyDesignBadge({ design }: { design: StudyDesignType }) {
  const labelMap: Record<StudyDesignType, string> = {
    INTERVENTIONAL_TRIAL: "RCT / Interventional",
    OBSERVATIONAL_STUDY: "Observational Study",
    PREDICTION_MODEL: "Prediction Model",
    SYSTEMATIC_REVIEW_META_ANALYSIS: "Systematic Review",
    MECHANISTIC_EXPLORATORY: "Mechanistic / Lab",
    UNKNOWN: "Study",
  };

  const variantMap: Record<StudyDesignType, BadgeProps["variant"]> = {
    INTERVENTIONAL_TRIAL: "supports",
    OBSERVATIONAL_STUDY: "incomparable",
    PREDICTION_MODEL: "qualifies",
    SYSTEMATIC_REVIEW_META_ANALYSIS: "accent",
    MECHANISTIC_EXPLORATORY: "default",
    UNKNOWN: "default",
  };

  return (
    <Badge variant={variantMap[design] || "default"} size="sm">
      {labelMap[design] || design}
    </Badge>
  );
}

export function SubclaimBadge({ status }: { status: SubclaimStatus }) {
  const variant: BadgeProps["variant"] =
    status === "SUPPORTED"
      ? "supports"
      : status === "QUALIFIED"
      ? "qualifies"
      : "contradicts";

  return <Badge variant={variant} size="sm">{status}</Badge>;
}
