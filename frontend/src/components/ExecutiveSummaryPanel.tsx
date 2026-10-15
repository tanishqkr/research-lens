"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ResearchReport, SubclaimAssessment, DisagreementExplanation } from "@/types/scientific";
import { Badge, SubclaimBadge } from "./ui/Badge";
import {
  Lightbulb,
  CheckCircle,
  AlertTriangle,
  HelpCircle,
  Scale,
  ShieldAlert,
  BarChart3,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface ExecutiveSummaryPanelProps {
  report?: ResearchReport | null;
}

export function ExecutiveSummaryPanel({ report }: ExecutiveSummaryPanelProps) {
  const [showDisagreements, setShowDisagreements] = useState(true);

  if (!report || !report.executive_summary) return null;
  const es = report.executive_summary;

  // Format overall assessment color
  const assessmentLower = (es.overall_evidence_assessment || "").toLowerCase();
  const assessmentVariant =
    assessmentLower.includes("support")
      ? "supports"
      : assessmentLower.includes("mixed") || assessmentLower.includes("context")
      ? "qualifies"
      : assessmentLower.includes("challenge") || assessmentLower.includes("contradict")
      ? "contradicts"
      : "incomparable";

  const subclaims = es.subclaim_assessments?.length
    ? es.subclaim_assessments
    : report.subclaim_assessments || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="w-full rounded-xl border border-border bg-surface shadow-card overflow-hidden transition-all"
    >
      {/* Header bar with Intent and State */}
      <div className="p-5 sm:p-6 border-b border-border bg-surface-subtle/40 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-ink tracking-tight flex items-center gap-2">
              <Scale className="w-5 h-5 text-accent" />
              Executive Research Summary
            </h2>
            <Badge variant="outline" size="sm">
              {report.query_intent}
            </Badge>
          </div>
          <p className="text-xs text-ink-muted mt-0.5">
            Condition-aware synthesis grounded in peer-reviewed scholarly evidence
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge
            variant={
              report.execution_state === "COMPLETED"
                ? "supports"
                : report.execution_state === "COMPLETED_WITH_LIMITATIONS"
                ? "qualifies"
                : "insufficient"
            }
          >
            {report.execution_state}
          </Badge>
          {report.run_id && (
            <span className="text-[11px] font-mono text-ink-subtle hidden sm:inline">
              Run: {report.run_id.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      <div className="p-5 sm:p-6 space-y-6">
        {/* Bottom Line Statement */}
        <div className="rounded-lg border border-accent-border/60 bg-accent-surface/50 p-4 sm:p-5">
          <div className="flex items-start gap-3">
            <div className="p-1.5 rounded-md bg-accent text-white mt-0.5">
              <Lightbulb className="w-4 h-4" />
            </div>
            <div className="space-y-2 flex-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-accent-text">
                  Scientific Bottom Line
                </span>
                <Badge variant={assessmentVariant} size="sm">
                  {String(es.overall_evidence_assessment || "EVALUATED").toUpperCase()}
                </Badge>
              </div>
              <p className="text-sm sm:text-base font-medium text-ink leading-relaxed">
                {es.bottom_line}
              </p>
            </div>
          </div>
        </div>

        {/* Decomposed Subclaims Assessment Matrix */}
        {subclaims.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-muted flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-accent" />
              Independent Subclaim Evidence Assessments
            </h3>
            <div className="grid grid-cols-1 gap-2.5">
              {subclaims.map((sc, idx) => (
                <div
                  key={idx}
                  className="rounded-lg border border-border bg-surface-subtle p-3.5 text-xs space-y-1.5 hover:border-border-strong transition-colors"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-semibold text-ink text-sm">
                      {sc.subclaim_text}
                    </span>
                    <SubclaimBadge status={sc.status} />
                  </div>
                  <p className="text-ink-muted leading-relaxed">
                    {sc.rationale}
                  </p>
                  {sc.evidence_paper_ids && sc.evidence_paper_ids.length > 0 && (
                    <div className="text-[11px] text-ink-subtle font-mono pt-1">
                      Referenced Evidence: {sc.evidence_paper_ids.join(", ")}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Split View: Strongest Supporting vs Strongest Conflicting */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-lg border border-evidence-supports-border/70 bg-evidence-supports-bg/40 p-4 space-y-2">
            <div className="flex items-center gap-1.5 text-evidence-supports-text font-semibold text-xs uppercase tracking-wider">
              <CheckCircle className="w-4 h-4" />
              Leading Supporting Evidence
            </div>
            <p className="text-xs text-ink leading-relaxed">
              {es.strongest_supporting_evidence}
            </p>
          </div>

          <div className="rounded-lg border border-evidence-contradicts-border/70 bg-evidence-contradicts-bg/40 p-4 space-y-2">
            <div className="flex items-center gap-1.5 text-evidence-contradicts-text font-semibold text-xs uppercase tracking-wider">
              <AlertTriangle className="w-4 h-4" />
              Conflicting / Qualifying Evidence
            </div>
            <p className="text-xs text-ink leading-relaxed">
              {es.strongest_conflicting_evidence}
            </p>
          </div>
        </div>

        {/* Why Studies Disagree (Moderator & Context Analysis) */}
        <div className="rounded-lg border border-border bg-surface-subtle overflow-hidden">
          <motion.button
            whileTap={{ scale: 0.995 }}
            onClick={() => setShowDisagreements(!showDisagreements)}
            className="w-full p-4 flex items-center justify-between text-left hover:bg-surface-hover/50 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-accent" />
              <span className="text-sm font-semibold text-ink">
                Why Studies Disagree (Moderator & Context Analysis)
              </span>
            </div>
            {showDisagreements ? (
              <ChevronUp className="w-4 h-4 text-ink-muted" />
            ) : (
              <ChevronDown className="w-4 h-4 text-ink-muted" />
            )}
          </motion.button>

          {showDisagreements && (
            <div className="p-4 pt-1 border-t border-border space-y-4 text-xs">
              <p className="text-ink leading-relaxed">
                {es.why_studies_disagree}
              </p>

              {report.disagreements && report.disagreements.length > 0 && (
                <div className="space-y-3 pt-2">
                  <div className="font-semibold text-ink uppercase tracking-wider text-[11px]">
                    Specific Disagreement Dissections
                  </div>
                  <div className="space-y-2">
                    {report.disagreements.map((d, i) => (
                      <div
                        key={i}
                        className="p-3 rounded-md border border-border bg-surface space-y-1.5"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-1">
                          <span className="font-semibold text-ink">
                            {d.topic_or_metric}
                          </span>
                          <Badge
                            variant={
                              d.verification_status === "SUPPORTED"
                                ? "supports"
                                : d.verification_status === "PLAUSIBLE_UNVERIFIED"
                                ? "qualifies"
                                : "insufficient"
                            }
                            size="sm"
                          >
                            {d.verification_status}
                          </Badge>
                        </div>
                        <p className="text-ink-muted">
                          <strong>Plausible moderator:</strong> {d.plausible_moderator_or_reason}
                        </p>
                        <div className="text-[11px] text-ink-subtle">
                          Disputing studies: {d.studies_in_disagreement.join(", ")}
                        </div>
                        {d.suggested_follow_up_investigation && (
                          <div className="text-[11px] text-accent-text pt-0.5 italic">
                            Suggested follow-up: {d.suggested_follow_up_investigation}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Robustness, Limitations & Research Coverage Footer */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-border">
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-ink-muted uppercase tracking-wider text-[11px]">
              <ShieldAlert className="w-3.5 h-3.5 text-evidence-qualifies-text" />
              Evidence Robustness & Limitations
            </div>
            <p className="text-ink-muted leading-relaxed">
              {es.confidence_and_limitations}
            </p>
          </div>

          <div className="space-y-1.5 text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-ink-muted uppercase tracking-wider text-[11px]">
              <BarChart3 className="w-3.5 h-3.5 text-accent" />
              Research Coverage & Completion
            </div>
            <p className="text-ink-muted leading-relaxed">
              {es.research_coverage}
            </p>
            <div className="text-[11px] text-ink-subtle italic">
              Status: {report.stopped_reason}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
