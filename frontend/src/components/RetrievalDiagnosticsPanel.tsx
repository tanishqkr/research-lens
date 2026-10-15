"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  RetrievalDiagnostics,
  ResearchReport,
  PaperTriageRecord,
} from "@/types/scientific";
import { Badge } from "./ui/Badge";
import {
  Activity,
  Database,
  Search,
  Filter,
  FileCheck2,
  FileText,
  AlertCircle,
  CheckCircle,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Cpu,
  ShieldCheck,
  Zap,
} from "lucide-react";

interface RetrievalDiagnosticsPanelProps {
  diagnostics?: RetrievalDiagnostics | null;
  report?: ResearchReport | null;
}

export function RetrievalDiagnosticsPanel({
  diagnostics,
  report,
}: RetrievalDiagnosticsPanelProps) {
  const [showTriageRecords, setShowTriageRecords] = useState(false);
  const [triageFilter, setTriageFilter] = useState<string>("ALL");

  const diag = diagnostics || {};
  const rep = report || ({} as Partial<ResearchReport>);
  const triageRecords = rep.triage_records || [];

  const filteredTriage = triageRecords.filter((rec) => {
    if (triageFilter === "ALL") return true;
    if (triageFilter === "SELECTED") return rec.selected_for_deep_dive;
    if (triageFilter === "FALLBACK") return rec.is_heuristic_fallback;
    return rec.category === triageFilter;
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 w-full"
    >
      {/* Overview Header */}
      <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 shadow-card space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-4">
          <div>
            <h3 className="text-base font-bold text-ink flex items-center gap-2">
              <Activity className="w-5 h-5 text-accent" />
              Retrieval & Pipeline Diagnostics
            </h3>
            <p className="text-xs text-ink-muted mt-0.5">
              Multimodal provenance audit across OpenAlex, arXiv, and empirical triage stages
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-ink-subtle font-mono">
              State: <strong className="text-ink">{rep.execution_state || diag.execution_state || "COMPLETED"}</strong>
            </span>
          </div>
        </div>

        {/* Scholarly Providers Status Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          {/* OpenAlex */}
          <div className="rounded-lg border border-border bg-surface-subtle p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-accent" />
                <span className="text-xs font-bold text-ink">OpenAlex</span>
              </div>
              <Badge
                variant={
                  diag.openalex?.status === 200 ||
                  diag.openalex?.status === "ok" ||
                  !diag.openalex?.error ||
                  diag.openalex?.error === "OK" ||
                  diag.openalex?.error === "CACHED"
                    ? "supports"
                    : "contradicts"
                }
                size="sm"
              >
                {diag.openalex?.status === 200
                  ? "ONLINE"
                  : String(diag.openalex?.status ?? "OK").toUpperCase()}
              </Badge>
            </div>
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-ink-muted">Papers Retrieved:</span>
              <span className="font-mono font-bold text-ink text-sm">
                {diag.openalex?.results_count ?? 0}
              </span>
            </div>
            {diag.openalex?.error &&
              diag.openalex.error !== "OK" &&
              diag.openalex.error !== "CACHED" && (
                <p className="text-[11px] text-evidence-contradicts-text">
                  {diag.openalex.error}
                </p>
              )}
          </div>

          {/* arXiv */}
          <div className="rounded-lg border border-border bg-surface-subtle p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-accent" />
                <span className="text-xs font-bold text-ink">arXiv Preprints</span>
              </div>
              <Badge
                variant={
                  diag.arxiv?.status === 200 ||
                  diag.arxiv?.status === "ok" ||
                  !diag.arxiv?.error ||
                  diag.arxiv?.error === "OK" ||
                  diag.arxiv?.error === "CACHED"
                    ? "supports"
                    : "contradicts"
                }
                size="sm"
              >
                {diag.arxiv?.status === 200
                  ? "ONLINE"
                  : String(diag.arxiv?.status ?? "OK").toUpperCase()}
              </Badge>
            </div>
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-ink-muted">Papers Retrieved:</span>
              <span className="font-mono font-bold text-ink text-sm">
                {diag.arxiv?.results_count ?? 0}
              </span>
            </div>
            {diag.arxiv?.error &&
              diag.arxiv.error !== "OK" &&
              diag.arxiv.error !== "CACHED" && (
                <p className="text-[11px] text-evidence-contradicts-text">
                  {diag.arxiv.error}
                </p>
              )}
          </div>

          {/* Groq / Structured Interpretation */}
          <div className="rounded-lg border border-border bg-surface-subtle p-3.5 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-accent" />
                <span className="text-xs font-bold text-ink">Groq LLM Auditor</span>
              </div>
              <Badge
                variant={diag.groq_assisted ? "accent" : "default"}
                size="sm"
              >
                {diag.groq_assisted ? "ACTIVE" : "FALLBACK"}
              </Badge>
            </div>
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-ink-muted">Audit Mode:</span>
              <span className="font-medium text-ink">
                {diag.groq_assisted ? "AI Assisted" : "Deterministic Rules"}
              </span>
            </div>
          </div>
        </div>

        {/* Literature Funnel & Full-Text Coverage Metrics */}
        <div className="space-y-2 pt-2 border-t border-border">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-subtle flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-accent" />
            Empirical Screening Funnel & Corpus Coverage
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2.5 text-xs">
            <div className="p-3 rounded-lg border border-border bg-surface-subtle text-center space-y-1">
              <div className="text-ink-muted text-[11px]">Total Retrieved</div>
              <div className="font-mono text-base font-bold text-ink">
                {rep.candidate_papers_retrieved || diag.raw_total || 0}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-border bg-surface-subtle text-center space-y-1">
              <div className="text-ink-muted text-[11px]">After Dedup</div>
              <div className="font-mono text-base font-bold text-ink">
                {diag.after_dedup || rep.papers_triaged || 0}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-border bg-surface-subtle text-center space-y-1">
              <div className="text-ink-muted text-[11px]">Triaged Relevant</div>
              <div className="font-mono text-base font-bold text-accent-text">
                {rep.papers_triaged || 0}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-border bg-surface-subtle text-center space-y-1">
              <div className="text-ink-muted text-[11px]">Deep-Dived</div>
              <div className="font-mono text-base font-bold text-accent-text">
                {rep.papers_deep_dived || 0}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-border bg-surface-subtle text-center space-y-1">
              <div className="text-ink-muted text-[11px]">Full-Text Used</div>
              <div className="font-mono text-base font-bold text-evidence-supports-text">
                {rep.full_texts_accessible || 0}
              </div>
            </div>

            <div className="p-3 rounded-lg border border-border bg-surface-subtle text-center space-y-1">
              <div className="text-ink-muted text-[11px]">Comparisons</div>
              <div className="font-mono text-base font-bold text-ink">
                {rep.comparisons_evaluated || 0}
              </div>
            </div>
          </div>
        </div>

        {/* Groq Structured PICO Breakdown (if available) */}
        {diag.groq_structured && (
          <div className="rounded-lg border border-accent-border/60 bg-accent-surface/40 p-4 space-y-2 text-xs">
            <div className="font-semibold text-accent-text flex items-center gap-1.5 uppercase tracking-wider text-[11px]">
              <Zap className="w-3.5 h-3.5" />
              Groq Query Decomposition Analysis
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-ink">
              <div>
                <strong className="text-ink-muted block text-[10px] uppercase">Population:</strong>
                <span>{diag.groq_structured.population || "—"}</span>
              </div>
              <div>
                <strong className="text-ink-muted block text-[10px] uppercase">Intervention:</strong>
                <span>{diag.groq_structured.intervention || "—"}</span>
              </div>
              <div>
                <strong className="text-ink-muted block text-[10px] uppercase">Comparator:</strong>
                <span>{diag.groq_structured.comparator || "—"}</span>
              </div>
              <div>
                <strong className="text-ink-muted block text-[10px] uppercase">Outcome:</strong>
                <span>{diag.groq_structured.outcome || "—"}</span>
              </div>
            </div>

            {diag.groq_structured.search_keywords && (
              <div className="pt-1 text-[11px] text-ink-muted flex items-center gap-1.5 flex-wrap">
                <span className="font-semibold text-ink">Generated Keywords:</span>
                {diag.groq_structured.search_keywords.map((kw, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 rounded bg-surface border border-border font-mono text-[10px]"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Dispatched Search Queries List */}
        {diag.search_queries_dispatched &&
          diag.search_queries_dispatched.length > 0 && (
            <div className="space-y-1.5 pt-1 text-xs">
              <div className="font-semibold text-ink-muted uppercase tracking-wider text-[11px] flex items-center gap-1.5">
                <Search className="w-3.5 h-3.5 text-accent" />
                Scholarly API Queries Dispatched
              </div>
              <ul className="space-y-1 font-mono text-[11px]">
                {diag.search_queries_dispatched.map((q, idx) => (
                  <li
                    key={idx}
                    className="p-2 rounded bg-surface-subtle border border-border text-ink truncate"
                  >
                    {q}
                  </li>
                ))}
              </ul>
            </div>
          )}
      </div>

      {/* Paper Triage Breakdown Section */}
      {triageRecords.length > 0 && (
        <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 shadow-card space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h4 className="text-sm font-bold text-ink flex items-center gap-2">
                <Filter className="w-4 h-4 text-accent" />
                Paper Triage & Relevance Classification Log
              </h4>
              <p className="text-xs text-ink-muted mt-0.5">
                Audit record showing classification category and rationale for each paper
              </p>
            </div>

            <div className="flex items-center gap-2">
              <motion.button
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowTriageRecords(!showTriageRecords)}
                className="text-xs text-accent hover:text-accent-hover font-semibold flex items-center gap-1 cursor-pointer"
              >
                {showTriageRecords ? (
                  <>
                    <ChevronUp className="w-4 h-4" />
                    <span>Hide ({triageRecords.length}) Records</span>
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-4 h-4" />
                    <span>View ({triageRecords.length}) Records</span>
                  </>
                )}
              </motion.button>
            </div>
          </div>

          {showTriageRecords && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="space-y-3 pt-2 border-t border-border"
            >
              {/* Quick triage filter pills */}
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                {[
                  { id: "ALL", label: "All Papers" },
                  { id: "SELECTED", label: "Deep-Dived Only" },
                  { id: "DIRECT_COMPARISON", label: "Direct Comparison" },
                  { id: "DIRECT_EMPIRICAL", label: "Direct Empirical" },
                  { id: "INDIRECT_EVIDENCE", label: "Indirect Evidence" },
                  { id: "REVIEW_SYNTHESIS", label: "Review / Synthesis" },
                  { id: "FALLBACK", label: "Heuristic Fallback" },
                ].map((b) => (
                  <motion.button
                    key={b.id}
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => setTriageFilter(b.id)}
                    className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
                      triageFilter === b.id
                        ? "bg-accent text-white font-semibold shadow-sm"
                        : "bg-surface-subtle text-ink-muted hover:text-ink border border-border"
                    }`}
                  >
                    {b.label}
                  </motion.button>
                ))}
              </div>

              {/* Triage items list */}
              <div className="space-y-2.5">
                {filteredTriage.map((rec) => (
                  <div
                    key={rec.paper_id}
                    className="p-3.5 rounded-lg border border-border bg-surface-subtle text-xs space-y-2 hover:border-border-strong transition-colors"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge
                          variant={
                            rec.category === "DIRECT_COMPARISON" || rec.category === "DIRECT_EMPIRICAL"
                              ? "supports"
                              : rec.category === "REVIEW_SYNTHESIS"
                              ? "accent"
                              : rec.category === "INDIRECT_EVIDENCE" || rec.category === "INDIRECT_EMPIRICAL"
                              ? "qualifies"
                              : "default"
                          }
                          size="sm"
                        >
                          {rec.category}
                        </Badge>

                        {rec.selected_for_deep_dive ? (
                          <span className="text-[10px] font-bold text-evidence-supports-text bg-evidence-supports-bg border border-evidence-supports-border px-1.5 py-0.5 rounded">
                            Deep-Dived
                          </span>
                        ) : (
                          <span className="text-[10px] text-ink-subtle px-1.5 py-0.5 rounded bg-surface border border-border">
                            Screened Only
                          </span>
                        )}

                        {rec.is_heuristic_fallback && (
                          <span className="text-[10px] text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/20 border border-amber-300 dark:border-amber-900/60 px-1.5 py-0.5 rounded">
                            Fallback Triage
                          </span>
                        )}
                      </div>

                      <span className="font-mono text-[10px] text-ink-subtle">
                        {rec.paper_id}
                      </span>
                    </div>

                    <div className="font-bold text-ink">{rec.title}</div>

                    <p className="text-ink-muted leading-relaxed">
                      <strong>Relevance:</strong> {rec.relevance_rationale}
                    </p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  );
}
