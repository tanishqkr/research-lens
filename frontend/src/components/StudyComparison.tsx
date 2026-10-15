"use client";

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  FindingComparisonItem,
  FindingPair,
  Prediction,
  RelationLabel,
} from "@/types/scientific";
import { Badge, RelationBadge, DirectionBadge, StudyDesignBadge } from "./ui/Badge";
import {
  GitCompare,
  Filter,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Split,
  ChevronDown,
  ChevronUp,
  Percent,
  Layers,
  ArrowRight,
  ShieldCheck,
  FileText,
  Quote,
} from "lucide-react";

interface StudyComparisonProps {
  comparisons: FindingComparisonItem[];
}

export function StudyComparison({ comparisons = [] }: StudyComparisonProps) {
  const [verdictFilter, setVerdictFilter] = useState<string>("ALL");
  const [expandedPairs, setExpandedPairs] = useState<Record<string, boolean>>({});

  const toggleExpand = (pairId: string) => {
    setExpandedPairs((prev) => ({
      ...prev,
      [pairId]: !prev[pairId],
    }));
  };

  const filteredComparisons = useMemo(() => {
    if (verdictFilter === "ALL") return comparisons;
    return comparisons.filter(
      (c) => c.prediction.predicted_relation === verdictFilter
    );
  }, [comparisons, verdictFilter]);

  // Metric counts for quick badges
  const counts = useMemo(() => {
    const c = {
      TOTAL: comparisons.length,
      SUPPORTS: 0,
      CONTRADICTS: 0,
      QUALIFIES: 0,
      INCOMPARABLE: 0,
      INSUFFICIENT: 0,
    };
    comparisons.forEach((item) => {
      const rel = item.prediction.predicted_relation;
      if (rel === "SUPPORTS_OR_CONSISTENT") c.SUPPORTS++;
      else if (rel === "CONTRADICTS") c.CONTRADICTS++;
      else if (rel === "QUALIFIES_OR_PARTIAL_DISAGREEMENT") c.QUALIFIES++;
      else if (rel === "INCOMPARABLE_CONDITIONS") c.INCOMPARABLE++;
      else if (rel === "INSUFFICIENT_EVIDENCE") c.INSUFFICIENT++;
    });
    return c;
  }, [comparisons]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-5 w-full"
    >
      {/* Header and Filter Control */}
      <div className="rounded-xl border border-border bg-surface p-4 sm:p-5 shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-accent" />
            <h3 className="text-base font-bold text-ink">
              Pairwise Study Comparison & Condition Disambiguation
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-surface-subtle border border-border text-ink-muted font-mono font-medium">
              {filteredComparisons.length} pairs evaluated
            </span>
          </div>

          <div className="text-xs text-ink-subtle">
            Evaluates 5-class relation between empirical findings
          </div>
        </div>

        {/* 5-Class Verdict Filter Buttons */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-border/70 text-xs">
          <div className="flex items-center gap-1.5 text-ink-muted mr-1">
            <Filter className="w-3.5 h-3.5 text-accent" />
            <span className="font-semibold uppercase tracking-wider text-[11px]">
              Verdict:
            </span>
          </div>

          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVerdictFilter("ALL")}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              verdictFilter === "ALL"
                ? "bg-accent text-white font-semibold shadow-sm"
                : "bg-surface-subtle text-ink-muted hover:text-ink hover:bg-surface-hover border border-border"
            }`}
          >
            All Verdicts ({counts.TOTAL})
          </motion.button>

          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVerdictFilter("CONTRADICTS")}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              verdictFilter === "CONTRADICTS"
                ? "bg-evidence-contradicts-bg text-evidence-contradicts-text border border-evidence-contradicts-border font-bold"
                : "bg-surface-subtle text-ink-muted hover:text-ink border border-border"
            }`}
          >
            Contradicts ({counts.CONTRADICTS})
          </motion.button>

          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVerdictFilter("INCOMPARABLE_CONDITIONS")}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              verdictFilter === "INCOMPARABLE_CONDITIONS"
                ? "bg-evidence-incomparable-bg text-evidence-incomparable-text border border-evidence-incomparable-border font-bold"
                : "bg-surface-subtle text-ink-muted hover:text-ink border border-border"
            }`}
          >
            Incomparable Conditions ({counts.INCOMPARABLE})
          </motion.button>

          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVerdictFilter("QUALIFIES_OR_PARTIAL_DISAGREEMENT")}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              verdictFilter === "QUALIFIES_OR_PARTIAL_DISAGREEMENT"
                ? "bg-evidence-qualifies-bg text-evidence-qualifies-text border border-evidence-qualifies-border font-bold"
                : "bg-surface-subtle text-ink-muted hover:text-ink border border-border"
            }`}
          >
            Qualifies / Partial ({counts.QUALIFIES})
          </motion.button>

          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVerdictFilter("SUPPORTS_OR_CONSISTENT")}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              verdictFilter === "SUPPORTS_OR_CONSISTENT"
                ? "bg-evidence-supports-bg text-evidence-supports-text border border-evidence-supports-border font-bold"
                : "bg-surface-subtle text-ink-muted hover:text-ink border border-border"
            }`}
          >
            Supports ({counts.SUPPORTS})
          </motion.button>

          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setVerdictFilter("INSUFFICIENT_EVIDENCE")}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              verdictFilter === "INSUFFICIENT_EVIDENCE"
                ? "bg-evidence-insufficient-bg text-evidence-insufficient-text border border-evidence-insufficient-border font-bold"
                : "bg-surface-subtle text-ink-muted hover:text-ink border border-border"
            }`}
          >
            Insufficient ({counts.INSUFFICIENT})
          </motion.button>
        </div>
      </div>

      {/* Comparisons List */}
      {comparisons.length === 0 ? (
        <div className="rounded-xl border border-border bg-surface p-8 text-center text-ink-muted text-sm space-y-2">
          <HelpCircle className="w-6 h-6 mx-auto text-ink-subtle" />
          <p className="font-medium text-ink">No pairwise comparisons evaluated for this query.</p>
          <p className="text-xs text-ink-muted">
            Condition-aware contradiction analysis requires at least two empirical findings from retrieved literature.
          </p>
        </div>
      ) : filteredComparisons.length === 0 ? (
        <div className="rounded-xl border border-border bg-surface p-8 text-center text-ink-muted text-sm space-y-2">
          <HelpCircle className="w-6 h-6 mx-auto text-ink-subtle" />
          <p>No study comparisons match the selected verdict filter.</p>
          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setVerdictFilter("ALL")}
            className="text-xs text-accent hover:underline font-medium cursor-pointer"
          >
            Show all comparisons
          </motion.button>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredComparisons.map((item, index) => {
            const { pair, prediction } = item;
            const pairId = pair?.pair_id || `pair-${index}`;
            const isExpanded = expandedPairs[pairId] ?? true;
            const diffs = prediction?.condition_differences;

            const compatPct = Math.max(
              0,
              Math.min(100, Math.round(((diffs?.overall_compatibility ?? 0.5) * 100)))
            );

            const hasDiscrepancyWithUnconditioned =
              prediction?.unconditioned_relation &&
              prediction.unconditioned_relation !== prediction.predicted_relation;

            return (
              <motion.div
                key={pairId}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
                whileHover={{ y: -2, transition: { duration: 0.15 } }}
                className="rounded-xl border border-border bg-surface shadow-card hover:border-border-strong transition-shadow overflow-hidden w-full"
              >
                {/* Comparison Card Top Bar */}
                <div className="p-4 sm:p-5 border-b border-border bg-surface-subtle/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <RelationBadge relation={prediction.predicted_relation} />

                      {prediction.provenance_verified && (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-evidence-supports-text bg-evidence-supports-bg border border-evidence-supports-border px-2 py-0.5 rounded">
                          <ShieldCheck className="w-3.5 h-3.5" />
                          Provenance Verified
                        </span>
                      )}

                      {prediction.confidence_score !== null &&
                        prediction.confidence_score !== undefined && (
                          <span className="text-[11px] text-ink-subtle font-mono px-2 py-0.5 rounded bg-surface border border-border">
                            Confidence: {(prediction.confidence_score * 100).toFixed(0)}%
                          </span>
                        )}
                    </div>

                    <div className="text-xs text-ink-muted pt-0.5">
                      Pair ID: <span className="font-mono text-ink">{pairId}</span>
                      {pair.domain && (
                        <span className="ml-2 font-medium text-accent-text">
                          • {pair.domain}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    {/* Overall Compatibility Meter */}
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-surface text-xs">
                      <span className="text-ink-muted font-medium">
                        Condition Compatibility:
                      </span>
                      <span
                        className={`font-mono font-bold ${
                          compatPct >= 70
                            ? "text-evidence-supports-text"
                            : compatPct >= 40
                            ? "text-evidence-qualifies-text"
                            : "text-evidence-contradicts-text"
                        }`}
                      >
                        {compatPct}%
                      </span>
                    </div>

                    <motion.button
                      whileTap={{ scale: 0.9 }}
                      onClick={() => toggleExpand(pairId)}
                      className="p-1.5 rounded-md hover:bg-surface-hover text-ink-muted transition-colors cursor-pointer"
                      aria-label={isExpanded ? "Collapse pair" : "Expand pair"}
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </motion.button>
                  </div>
                </div>

                {/* Main Content Area */}
                <div className="p-4 sm:p-5 space-y-4">
                  {/* Condition Disambiguation Banner (Unconditioned vs Conditioned) */}
                  {hasDiscrepancyWithUnconditioned && (
                    <div className="rounded-lg border border-evidence-qualifies-border bg-evidence-qualifies-bg p-3 text-xs space-y-1">
                      <div className="font-bold text-evidence-qualifies-text flex items-center gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        Condition Disambiguation Corrected False Classification
                      </div>
                      <p className="text-ink leading-relaxed">
                        A naive lexical/surface check labeled this pair as{" "}
                        <strong>{prediction.unconditioned_relation}</strong>. By isolating
                        differing condition slots (such as population or follow-up horizon),
                        the pipeline corrected the verdict to{" "}
                        <strong>{prediction.predicted_relation}</strong>.
                      </p>
                    </div>
                  )}

                  {/* Summary of Differences and Explanation */}
                  <div className="rounded-lg border border-border bg-surface-subtle p-3.5 space-y-2 text-xs">
                    <div className="font-semibold uppercase tracking-wider text-[11px] text-ink-subtle">
                      Pipeline Synthesis & Reason for Verdict
                    </div>
                    <p className="text-ink font-medium leading-relaxed">
                      {prediction.explanation}
                    </p>

                    {diffs?.summary_of_differences && (
                      <div className="pt-2 border-t border-border/60 text-ink-muted">
                        <strong className="text-ink">Condition Nuance:</strong>{" "}
                        {diffs.summary_of_differences}
                      </div>
                    )}
                  </div>

                  {/* Component Compatibility Breakdown */}
                  {diffs && (
                    <div className="space-y-1.5">
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-subtle flex items-center gap-1">
                        <Layers className="w-3 h-3 text-accent" />
                        Component-Wise Condition Similarity
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                        {(() => {
                          const iPct = Math.max(0, Math.min(100, Math.round((diffs.intervention_similarity ?? 0) * 100)));
                          const cPct = Math.max(0, Math.min(100, Math.round((diffs.comparator_similarity ?? 0) * 100)));
                          const oPct = Math.max(0, Math.min(100, Math.round((diffs.outcome_similarity ?? 0) * 100)));
                          const sPct = Math.max(0, Math.min(100, Math.round((diffs.setting_similarity ?? 0) * 100)));

                          return (
                            <>
                              <div className="p-2.5 rounded-lg border border-border bg-surface space-y-1">
                                <div className="flex items-center justify-between text-[11px] text-ink-muted">
                                  <span>Intervention</span>
                                  <span className="font-mono font-bold">{iPct}%</span>
                                </div>
                                <div className="w-full bg-surface-subtle rounded-full h-1.5 overflow-hidden">
                                  <div className="bg-accent h-1.5 rounded-full" style={{ width: `${iPct}%` }} />
                                </div>
                              </div>

                              <div className="p-2.5 rounded-lg border border-border bg-surface space-y-1">
                                <div className="flex items-center justify-between text-[11px] text-ink-muted">
                                  <span>Comparator</span>
                                  <span className="font-mono font-bold">{cPct}%</span>
                                </div>
                                <div className="w-full bg-surface-subtle rounded-full h-1.5 overflow-hidden">
                                  <div className="bg-accent h-1.5 rounded-full" style={{ width: `${cPct}%` }} />
                                </div>
                              </div>

                              <div className="p-2.5 rounded-lg border border-border bg-surface space-y-1">
                                <div className="flex items-center justify-between text-[11px] text-ink-muted">
                                  <span>Outcome</span>
                                  <span className="font-mono font-bold">{oPct}%</span>
                                </div>
                                <div className="w-full bg-surface-subtle rounded-full h-1.5 overflow-hidden">
                                  <div className="bg-accent h-1.5 rounded-full" style={{ width: `${oPct}%` }} />
                                </div>
                              </div>

                              <div className="p-2.5 rounded-lg border border-border bg-surface space-y-1">
                                <div className="flex items-center justify-between text-[11px] text-ink-muted">
                                  <span>Setting / Pop</span>
                                  <span className="font-mono font-bold">{sPct}%</span>
                                </div>
                                <div className="w-full bg-surface-subtle rounded-full h-1.5 overflow-hidden">
                                  <div className="bg-accent h-1.5 rounded-full" style={{ width: `${sPct}%` }} />
                                </div>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                    </div>
                  )}

                  {/* Side-by-side Findings: Finding A vs Finding B */}
                  {isExpanded && (
                    <div className="pt-2 space-y-2">
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-subtle">
                        Side-by-Side Findings Inspection
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Finding A */}
                        <div className="rounded-lg border border-border bg-surface-subtle/50 p-4 space-y-3">
                          <div className="flex items-center justify-between gap-2 border-b border-border/80 pb-2">
                            <span className="text-xs font-bold uppercase tracking-wider text-accent-text">
                              Finding A
                            </span>
                            <div className="flex items-center gap-1.5">
                              <DirectionBadge direction={pair.finding_a.direction} />
                              <StudyDesignBadge
                                design={pair.finding_a.conditions?.study_type || "UNKNOWN"}
                              />
                            </div>
                          </div>

                          <div>
                            <h5 className="text-xs font-bold text-ink leading-snug">
                              {pair.finding_a.paper_title}
                            </h5>
                            <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                              {pair.finding_a.finding_claim}
                            </p>
                          </div>

                          {/* Verbatim snippet */}
                          {pair.finding_a.evidence?.text && (
                            <div className="rounded border-l-2 border-accent bg-surface p-2.5 text-xs text-ink italic leading-relaxed">
                              &ldquo;{pair.finding_a.evidence.text}&rdquo;
                            </div>
                          )}

                          {/* Condition slots */}
                          <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-border/60">
                            <div>
                              <span className="text-ink-subtle block font-semibold">[I] Intervention:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_a.conditions?.intervention || "—"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-subtle block font-semibold">[C] Comparator:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_a.conditions?.comparator || "—"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-subtle block font-semibold">[O] Outcome:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_a.conditions?.outcome_metric || "—"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-subtle block font-semibold">[S] Setting:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_a.conditions?.population ||
                                  pair.finding_a.conditions?.dataset_setting ||
                                  "—"}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Finding B */}
                        <div className="rounded-lg border border-border bg-surface-subtle/50 p-4 space-y-3">
                          <div className="flex items-center justify-between gap-2 border-b border-border/80 pb-2">
                            <span className="text-xs font-bold uppercase tracking-wider text-accent-text">
                              Finding B
                            </span>
                            <div className="flex items-center gap-1.5">
                              <DirectionBadge direction={pair.finding_b.direction} />
                              <StudyDesignBadge
                                design={pair.finding_b.conditions?.study_type || "UNKNOWN"}
                              />
                            </div>
                          </div>

                          <div>
                            <h5 className="text-xs font-bold text-ink leading-snug">
                              {pair.finding_b.paper_title}
                            </h5>
                            <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                              {pair.finding_b.finding_claim}
                            </p>
                          </div>

                          {/* Verbatim snippet */}
                          {pair.finding_b.evidence?.text && (
                            <div className="rounded border-l-2 border-accent bg-surface p-2.5 text-xs text-ink italic leading-relaxed">
                              &ldquo;{pair.finding_b.evidence.text}&rdquo;
                            </div>
                          )}

                          {/* Condition slots */}
                          <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-border/60">
                            <div>
                              <span className="text-ink-subtle block font-semibold">[I] Intervention:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_b.conditions?.intervention || "—"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-subtle block font-semibold">[C] Comparator:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_b.conditions?.comparator || "—"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-subtle block font-semibold">[O] Outcome:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_b.conditions?.outcome_metric || "—"}
                              </span>
                            </div>
                            <div>
                              <span className="text-ink-subtle block font-semibold">[S] Setting:</span>
                              <span className="text-ink font-medium">
                                {pair.finding_b.conditions?.population ||
                                  pair.finding_b.conditions?.dataset_setting ||
                                  "—"}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}
