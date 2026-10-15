"use client";

import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Header } from "@/components/Header";
import { QueryComposer } from "@/components/QueryComposer";
import { SearchProgress } from "@/components/SearchProgress";
import { ExecutiveSummaryPanel } from "@/components/ExecutiveSummaryPanel";
import { EvidenceLedger } from "@/components/EvidenceLedger";
import { StudyComparison } from "@/components/StudyComparison";
import { RetrievalDiagnosticsPanel } from "@/components/RetrievalDiagnosticsPanel";
import { EmptyState } from "@/components/EmptyState";
import { BenchmarkExplorer } from "@/components/BenchmarkExplorer";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  AnalysisResponse,
  HealthResponse,
} from "@/types/scientific";
import { analyzeQuery, fetchHealth, fetchPresets } from "@/lib/api";
import {
  Scale,
  GitCompare,
  BookOpen,
  Activity,
  AlertCircle,
  RefreshCw,
  Sparkles,
  Layers,
  FileCheck2,
  FileText,
  Search,
} from "lucide-react";

export default function WorkbenchPage() {
  // Navigation & System state
  const [activeTab, setActiveTab] = useState<"search" | "benchmark">("search");
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [offlineMode, setOfflineMode] = useState<boolean>(false);
  const [presets, setPresets] = useState<string[]>([]);

  // Search Workbench state
  const [query, setQuery] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<{ message: string; detail?: string } | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);

  // Active section tab within search results
  const [activeResultView, setActiveResultView] = useState<
    "all" | "executive" | "comparisons" | "ledger" | "diagnostics"
  >("all");

  // Health check & Presets loading on mount
  useEffect(() => {
    let isMounted = true;

    async function checkSystem() {
      try {
        const health: HealthResponse = await fetchHealth();
        if (isMounted) {
          setIsBackendConnected(health.status === "healthy");
          setOfflineMode(!!health.offline_mode);
        }
      } catch {
        if (isMounted) {
          setIsBackendConnected(false);
        }
      }

      try {
        const p = await fetchPresets();
        if (isMounted) {
          setPresets(p);
        }
      } catch {
        // Fallback handled in api.ts
      }
    }

    checkSystem();

    return () => {
      isMounted = false;
    };
  }, []);

  // Analysis submission handler
  const handleAnalyzeQuery = useCallback(
    async (
      targetQuery: string,
      options: { limit: number; useGroq: boolean } = { limit: 10, useGroq: false }
    ) => {
      const q = targetQuery.trim();
      if (!q || isLoading) return;

      setIsLoading(true);
      setError(null);

      try {
        const res = await analyzeQuery(q, options);
        setAnalysisResult(res);
      } catch (err: any) {
        setError({
          message: err.message || "Failed to analyze scientific query.",
          detail: err.detail,
        });
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading]
  );

  const handleAnalyze = useCallback(
    (options: { limit: number; useGroq: boolean }) => {
      handleAnalyzeQuery(query, options);
    },
    [query, handleAnalyzeQuery]
  );

  const handleSelectQueryFromEmptyState = (selectedQuery: string, autoAnalyze?: boolean) => {
    setQuery(selectedQuery);
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (autoAnalyze) {
      handleAnalyzeQuery(selectedQuery, { limit: 10, useGroq: false });
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-workbench text-ink">
      {/* Top Application Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isBackendConnected={isBackendConnected}
        offlineMode={offlineMode}
      />

      {/* Main Workbench Body */}
      <main className="flex-1 w-full max-w-[1440px] px-6 md:px-12 lg:px-16 mx-auto py-6 space-y-6">
        {activeTab === "benchmark" ? (
          /* CAS-Con150 Benchmark Mode */
          <BenchmarkExplorer />
        ) : (
          /* Live Empirical Search Workbench */
          <div className="space-y-6">
            {/* Primary Query Composer */}
            <QueryComposer
              query={query}
              setQuery={setQuery}
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
              presets={presets}
            />

            {/* In-Flight Execution Progress */}
            {isLoading && <SearchProgress query={query} />}

            {/* Error Notification Alert */}
            {error && !isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
                className="rounded-xl border border-evidence-contradicts-border bg-evidence-contradicts-bg p-5 text-xs space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-bold text-evidence-contradicts-text text-sm">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>Analysis Error Encountered</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      onClick={() => handleAnalyze({ limit: 10, useGroq: false })}
                      variant="outline"
                      size="sm"
                      className="border-evidence-contradicts-border text-evidence-contradicts-text hover:bg-white/40"
                    >
                      <RefreshCw className="w-3.5 h-3.5 mr-1" />
                      Retry Analysis
                    </Button>
                    <Button
                      onClick={() => setError(null)}
                      variant="ghost"
                      size="sm"
                      className="text-evidence-contradicts-text hover:bg-white/40"
                    >
                      Dismiss
                    </Button>
                  </div>
                </div>
                <p className="text-ink font-medium">{error.message}</p>
                {error.detail && (
                  <div className="font-mono text-[11px] text-ink-muted bg-surface/60 p-2.5 rounded border border-border/50 break-words">
                    {error.detail}
                  </div>
                )}
              </motion.div>
            )}

            {/* Results Presentation Deck */}
            {analysisResult && !isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                {/* Metrics Summary Strip */}
                <div className="rounded-xl border border-border bg-surface p-4 sm:p-5 shadow-card flex flex-wrap items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-bold uppercase tracking-wider text-accent-text">
                        Empirical Synthesis Ready
                      </span>
                      <Badge variant="outline" size="sm">
                        Intent: {analysisResult.query_intent}
                      </Badge>
                      <Badge
                        variant={
                          analysisResult.execution_state === "COMPLETED"
                            ? "supports"
                            : analysisResult.execution_state === "COMPLETED_WITH_LIMITATIONS"
                            ? "qualifies"
                            : "default"
                        }
                        size="sm"
                      >
                        {analysisResult.execution_state}
                      </Badge>
                    </div>
                    <p className="text-xs text-ink-muted italic line-clamp-1">
                      &ldquo;{analysisResult.query}&rdquo;
                    </p>
                  </div>

                  {/* Summary Metric Counters */}
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <div className="text-right">
                      <div className="text-[10px] uppercase text-ink-subtle font-sans font-semibold">
                        Papers Screened
                      </div>
                      <div className="text-sm font-bold text-ink">
                        {analysisResult.papers_found}
                      </div>
                    </div>
                    <div className="text-border font-light">/</div>
                    <div className="text-right">
                      <div className="text-[10px] uppercase text-ink-subtle font-sans font-semibold">
                        Findings Extracted
                      </div>
                      <div className="text-sm font-bold text-accent-text">
                        {analysisResult.findings_extracted}
                      </div>
                    </div>
                    <div className="text-border font-light">/</div>
                    <div className="text-right">
                      <div className="text-[10px] uppercase text-ink-subtle font-sans font-semibold">
                        Pairs Evaluated
                      </div>
                      <div className="text-sm font-bold text-ink">
                        {analysisResult.comparisons?.length ?? 0}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Zero Papers Notice */}
                {analysisResult.papers_found === 0 && (
                  <div className="rounded-xl border border-evidence-qualifies-border bg-evidence-qualifies-bg p-5 text-xs space-y-2">
                    <div className="flex items-center gap-2 font-bold text-evidence-qualifies-text text-sm">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>Zero Empirical Papers Retrieved</span>
                    </div>
                    <p className="text-ink leading-relaxed">
                      No matching publications were retrieved across OpenAlex and arXiv for &ldquo;{analysisResult.query}&rdquo;.
                      Scholarly queries often succeed better with standard biomedical nomenclature, clinical acronym expansions, or broader search terms.
                    </p>
                    <div className="pt-2 flex flex-wrap items-center gap-2">
                      {presets.length > 0 && (
                        <Button
                          onClick={() => {
                            const p = presets[0];
                            setQuery(p);
                            handleAnalyzeQuery(p, { limit: 10, useGroq: false });
                          }}
                          variant="outline"
                          size="sm"
                        >
                          Try Curated Benchmark Preset
                        </Button>
                      )}
                      <Button
                        onClick={() => setActiveTab("benchmark")}
                        variant="secondary"
                        size="sm"
                      >
                        Switch to CAS-Con150 Explorer
                      </Button>
                    </div>
                  </div>
                )}

                {/* Section Navigation Tabs */}
                <div className="flex items-center gap-1.5 p-1 rounded-xl bg-surface-subtle border border-border overflow-x-auto text-xs font-medium">
                  <motion.button
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveResultView("all")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer whitespace-nowrap ${
                      activeResultView === "all"
                        ? "bg-surface text-ink font-bold shadow-sm"
                        : "text-ink-muted hover:text-ink"
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5 text-accent" />
                    <span>Complete Workbench</span>
                  </motion.button>

                  <motion.button
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveResultView("executive")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer whitespace-nowrap ${
                      activeResultView === "executive"
                        ? "bg-surface text-ink font-bold shadow-sm"
                        : "text-ink-muted hover:text-ink"
                    }`}
                  >
                    <Scale className="w-3.5 h-3.5 text-accent" />
                    <span>Executive Summary</span>
                  </motion.button>

                  <motion.button
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveResultView("comparisons")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer whitespace-nowrap ${
                      activeResultView === "comparisons"
                        ? "bg-surface text-ink font-bold shadow-sm"
                        : "text-ink-muted hover:text-ink"
                    }`}
                  >
                    <GitCompare className="w-3.5 h-3.5 text-accent" />
                    <span>Study Comparisons ({analysisResult.comparisons?.length ?? 0})</span>
                  </motion.button>

                  <motion.button
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveResultView("ledger")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer whitespace-nowrap ${
                      activeResultView === "ledger"
                        ? "bg-surface text-ink font-bold shadow-sm"
                        : "text-ink-muted hover:text-ink"
                    }`}
                  >
                    <BookOpen className="w-3.5 h-3.5 text-accent" />
                    <span>Evidence Ledger ({analysisResult.research_report?.evidence_ledger?.length || analysisResult.findings?.length || 0})</span>
                  </motion.button>

                  <motion.button
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveResultView("diagnostics")}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors cursor-pointer whitespace-nowrap ${
                      activeResultView === "diagnostics"
                        ? "bg-surface text-ink font-bold shadow-sm"
                        : "text-ink-muted hover:text-ink"
                    }`}
                  >
                    <Activity className="w-3.5 h-3.5 text-accent" />
                    <span>Retrieval Diagnostics</span>
                  </motion.button>
                </div>

                {/* View Sections */}
                {(activeResultView === "all" || activeResultView === "executive") && (
                  <section>
                    <ExecutiveSummaryPanel report={analysisResult.research_report} />
                  </section>
                )}

                {(activeResultView === "all" || activeResultView === "comparisons") && (
                  <section>
                    <StudyComparison comparisons={analysisResult.comparisons || []} />
                  </section>
                )}

                {(activeResultView === "all" || activeResultView === "ledger") && (
                  <section>
                    <EvidenceLedger
                      ledger={analysisResult.research_report?.evidence_ledger || []}
                      findings={analysisResult.findings || []}
                    />
                  </section>
                )}

                {(activeResultView === "all" || activeResultView === "diagnostics") && (
                  <section>
                    <RetrievalDiagnosticsPanel
                      diagnostics={analysisResult.diagnostics}
                      report={analysisResult.research_report}
                    />
                  </section>
                )}
              </motion.div>
            )}

            {/* Empty State when no query is currently analyzed */}
            {!analysisResult && !isLoading && (
              <EmptyState
                onSelectQuery={handleSelectQueryFromEmptyState}
                presets={presets}
              />
            )}
          </div>
        )}
      </main>

      {/* Academic Workbench Footer */}
      <footer className="border-t border-border bg-surface/60 py-6 text-xs text-ink-subtle">
        <div className="w-full max-w-[1440px] px-6 md:px-12 lg:px-16 mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-ink">Scientific Contradiction Finder</span>
            <span>•</span>
            <span>Condition-Aware Literature Workbench</span>
          </div>
          <div className="flex items-center gap-3">
            <span>OpenAlex Primary</span>
            <span>•</span>
            <span>arXiv Preprints</span>
            <span>•</span>
            <span>CAS-Con150 Gold Standard</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
