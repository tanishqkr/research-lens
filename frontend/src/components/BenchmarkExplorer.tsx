"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FindingPair, Prediction } from "@/types/scientific";
import { fetchBenchmarkPairs, classifyPair } from "@/lib/api";
import { Badge, RelationBadge, DirectionBadge, StudyDesignBadge } from "./ui/Badge";
import { Button } from "./ui/Button";
import {
  GitCompare,
  Database,
  Search,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Play,
  Layers,
  ArrowRight,
  ShieldCheck,
  Loader2,
} from "lucide-react";

export function BenchmarkExplorer() {
  const [pairs, setPairs] = useState<FindingPair[]>([]);
  const [selectedPairIndex, setSelectedPairIndex] = useState<number>(0);
  const [isLoadingPairs, setIsLoadingPairs] = useState<boolean>(true);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState<string>("");

  useEffect(() => {
    async function loadPairs() {
      try {
        setIsLoadingPairs(true);
        const res = await fetchBenchmarkPairs();
        setPairs(res.pairs || []);
      } catch (err: any) {
        setError(err.message || "Failed to load CAS-Con150 benchmark pairs");
      } finally {
        setIsLoadingPairs(false);
      }
    }
    loadPairs();
  }, []);

  const selectedPair = pairs[selectedPairIndex];

  // Auto classify or reset prediction on pair change
  useEffect(() => {
    setPrediction(null);
  }, [selectedPairIndex]);

  const handleClassify = async () => {
    if (!selectedPair) return;
    try {
      setIsClassifying(true);
      setError(null);
      const pred = await classifyPair(selectedPair);
      setPrediction(pred);
    } catch (err: any) {
      setError(err.message || "Failed to classify finding pair");
    } finally {
      setIsClassifying(false);
    }
  };

  const filteredPairs = pairs.filter((p, idx) => {
    if (!searchFilter.trim()) return true;
    const q = searchFilter.toLowerCase();
    const titleA = p.finding_a?.paper_title?.toLowerCase() || "";
    const titleB = p.finding_b?.paper_title?.toLowerCase() || "";
    const domain = p.domain?.toLowerCase() || "";
    const id = p.pair_id?.toLowerCase() || "";
    return (
      titleA.includes(q) ||
      titleB.includes(q) ||
      domain.includes(q) ||
      id.includes(q)
    );
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Benchmark Explorer Banner */}
      <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 shadow-card space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-ink tracking-tight flex items-center gap-2">
                <Database className="w-5 h-5 text-accent" />
                CAS-Con150 Benchmark Explorer
              </h2>
              <Badge variant="accent" size="sm">
                Condition-Aware Benchmark
              </Badge>
            </div>
            <p className="text-xs text-ink-muted mt-1">
              Explore 150 gold-standard scientific claim pairs across clinical medicine,
              epidemiology, biology, and machine learning.
            </p>
          </div>

          <div className="text-xs font-mono text-ink-subtle">
            Total Pairs: <strong className="text-ink">{pairs.length}</strong>
          </div>
        </div>
      </div>

      {isLoadingPairs ? (
        <div className="rounded-xl border border-border bg-surface p-12 text-center space-y-3 shadow-card">
          <Loader2 className="w-6 h-6 mx-auto text-accent animate-spin" />
          <p className="text-xs text-ink-muted">Loading benchmark dataset from backend...</p>
        </div>
      ) : error && pairs.length === 0 ? (
        <div className="rounded-xl border border-evidence-contradicts-border bg-evidence-contradicts-bg/30 p-6 text-center space-y-2">
          <AlertTriangle className="w-5 h-5 mx-auto text-evidence-contradicts-text" />
          <p className="text-xs font-medium text-evidence-contradicts-text">{error}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Pairs Selection Sidebar */}
          <div className="lg:col-span-4 space-y-3">
            <div className="rounded-xl border border-border bg-surface p-4 shadow-card space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-ink uppercase tracking-wider text-[11px]">
                  Select Benchmark Pair
                </span>
                <span className="text-ink-subtle font-mono">
                  {filteredPairs.length} matches
                </span>
              </div>

              <div className="relative">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-ink-subtle" />
                <input
                  type="text"
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  placeholder="Filter by title, domain, or ID..."
                  className="w-full text-xs rounded-md border border-border bg-surface-subtle pl-8 pr-2.5 py-1.5 text-ink placeholder:text-ink-subtle focus:outline-none focus:ring-1 focus:ring-accent"
                />
              </div>

              {/* Scrollable pairs list */}
              <div className="space-y-1.5 max-h-[560px] overflow-y-auto pr-1">
                {filteredPairs.length === 0 ? (
                  <div className="p-6 text-center text-xs text-ink-muted border border-dashed border-border rounded-lg">
                    No benchmark pairs match &ldquo;{searchFilter}&rdquo;
                  </div>
                ) : (
                  filteredPairs.map((p, idx) => {
                    const originalIdx = pairs.indexOf(p);
                    const isSelected = originalIdx === selectedPairIndex;

                    return (
                      <motion.button
                        key={p.pair_id || idx}
                        whileHover={{ y: -1 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedPairIndex(originalIdx)}
                        className={`w-full text-left p-3 rounded-lg border text-xs transition-all cursor-pointer ${
                          isSelected
                            ? "bg-accent-surface border-accent-border text-ink shadow-sm"
                            : "border-border bg-surface-subtle hover:bg-surface-hover text-ink-muted"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1 mb-1">
                          <span className="font-mono text-[10px] text-ink-subtle">
                            {p.pair_id}
                          </span>
                          {p.ground_truth_relation && (
                            <RelationBadge relation={p.ground_truth_relation} />
                          )}
                        </div>

                        <div className="font-semibold text-ink line-clamp-1 text-xs">
                          {p.finding_a?.paper_title || "Study A"}
                        </div>
                        <div className="text-[11px] text-ink-subtle line-clamp-1">
                          vs {p.finding_b?.paper_title || "Study B"}
                        </div>

                        {p.domain && (
                          <div className="mt-1 text-[10px] font-medium text-accent-text">
                            Domain: {p.domain}
                          </div>
                        )}
                      </motion.button>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Selected Pair Detail & Classifier Run */}
          <div className="lg:col-span-8 space-y-4">
            {!selectedPair ? (
              <div className="rounded-xl border border-border bg-surface p-12 text-center text-ink-muted text-xs space-y-2 shadow-card">
                <GitCompare className="w-6 h-6 mx-auto text-ink-subtle" />
                <p className="font-medium text-ink">No Pair Selected</p>
                <p>Select a benchmark pair from the list to view study details and run condition-aware classification.</p>
              </div>
            ) : (
              <div className="rounded-xl border border-border bg-surface p-5 sm:p-6 shadow-card space-y-5">
                {/* Header with Classification Action */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-ink">
                        {selectedPair.pair_id}
                      </span>
                      {selectedPair.domain && (
                        <span className="text-xs px-2 py-0.5 rounded bg-surface-subtle border border-border font-medium text-ink-muted">
                          {selectedPair.domain}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 pt-1 text-xs">
                      <span className="text-ink-muted">Gold-Standard Ground Truth:</span>
                      {selectedPair.ground_truth_relation ? (
                        <RelationBadge relation={selectedPair.ground_truth_relation} />
                      ) : (
                        <span className="text-ink-subtle">Unlabeled</span>
                      )}
                    </div>
                  </div>

                  <Button
                    onClick={handleClassify}
                    isLoading={isClassifying}
                    variant="primary"
                    size="sm"
                    className="self-start sm:self-center"
                  >
                    <Play className="w-3.5 h-3.5" />
                    Run Condition-Aware Classification
                  </Button>
                </div>

                {/* Prediction Output if evaluated */}
                {prediction && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                    className="rounded-xl border border-border bg-surface-subtle p-4 space-y-3.5"
                  >
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-2.5">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-ink uppercase tracking-wider">
                          Pipeline Classification:
                        </span>
                        <RelationBadge relation={prediction.predicted_relation} />
                      </div>

                      {/* Agreement with ground truth */}
                      {selectedPair.ground_truth_relation && (
                        <div className="flex items-center gap-1.5 text-xs font-semibold">
                          {prediction.predicted_relation === selectedPair.ground_truth_relation ? (
                            <span className="text-evidence-supports-text flex items-center gap-1">
                              <CheckCircle2 className="w-4 h-4" />
                              Matches Ground Truth
                            </span>
                          ) : (
                            <span className="text-evidence-contradicts-text flex items-center gap-1">
                              <XCircle className="w-4 h-4" />
                              Differs from Ground Truth
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    <p className="text-xs text-ink leading-relaxed font-medium">
                      {prediction.explanation}
                    </p>

                    {prediction.condition_differences && (
                      <div className="space-y-2 pt-2 border-t border-border">
                        <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-subtle">
                          Condition Differences & Compatibility
                        </div>
                        <p className="text-xs text-ink-muted">
                          {prediction.condition_differences.summary_of_differences}
                        </p>
                        <div className="text-xs text-ink">
                          Overall Compatibility:{" "}
                          <strong className="font-mono text-accent-text">
                            {(prediction.condition_differences.overall_compatibility * 100).toFixed(0)}%
                          </strong>
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* Side-by-side Inspection */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Finding A */}
                  <div className="rounded-lg border border-border bg-surface-subtle p-4 space-y-3">
                    <div className="flex items-center justify-between gap-2 border-b border-border pb-2">
                      <span className="text-xs font-bold uppercase tracking-wider text-accent-text">
                        Finding A
                      </span>
                      <div className="flex items-center gap-1.5">
                        <DirectionBadge direction={selectedPair.finding_a.direction} />
                        <StudyDesignBadge
                          design={selectedPair.finding_a.conditions?.study_type || "UNKNOWN"}
                        />
                      </div>
                    </div>

                    <div>
                      <h4 className="text-xs font-bold text-ink leading-snug">
                        {selectedPair.finding_a.paper_title}
                      </h4>
                      <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                        {selectedPair.finding_a.finding_claim}
                      </p>
                    </div>

                    {selectedPair.finding_a.evidence?.text && (
                      <div className="rounded border-l-2 border-accent bg-surface p-2.5 text-xs text-ink italic leading-relaxed">
                        &ldquo;{selectedPair.finding_a.evidence.text}&rdquo;
                      </div>
                    )}

                    {/* Slots */}
                    <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-border/60">
                      <div>
                        <span className="text-ink-subtle block font-semibold">[I] Intervention:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_a.conditions?.intervention || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-ink-subtle block font-semibold">[C] Comparator:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_a.conditions?.comparator || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-ink-subtle block font-semibold">[O] Outcome:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_a.conditions?.outcome_metric || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-ink-subtle block font-semibold">[S] Setting:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_a.conditions?.population ||
                            selectedPair.finding_a.conditions?.dataset_setting ||
                            "—"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Finding B */}
                  <div className="rounded-lg border border-border bg-surface-subtle p-4 space-y-3">
                    <div className="flex items-center justify-between gap-2 border-b border-border pb-2">
                      <span className="text-xs font-bold uppercase tracking-wider text-accent-text">
                        Finding B
                      </span>
                      <div className="flex items-center gap-1.5">
                        <DirectionBadge direction={selectedPair.finding_b.direction} />
                        <StudyDesignBadge
                          design={selectedPair.finding_b.conditions?.study_type || "UNKNOWN"}
                        />
                      </div>
                    </div>

                    <div>
                      <h4 className="text-xs font-bold text-ink leading-snug">
                        {selectedPair.finding_b.paper_title}
                      </h4>
                      <p className="text-xs text-ink-muted mt-1 leading-relaxed">
                        {selectedPair.finding_b.finding_claim}
                      </p>
                    </div>

                    {selectedPair.finding_b.evidence?.text && (
                      <div className="rounded border-l-2 border-accent bg-surface p-2.5 text-xs text-ink italic leading-relaxed">
                        &ldquo;{selectedPair.finding_b.evidence.text}&rdquo;
                      </div>
                    )}

                    {/* Slots */}
                    <div className="grid grid-cols-2 gap-2 text-[11px] pt-1 border-t border-border/60">
                      <div>
                        <span className="text-ink-subtle block font-semibold">[I] Intervention:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_b.conditions?.intervention || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-ink-subtle block font-semibold">[C] Comparator:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_b.conditions?.comparator || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-ink-subtle block font-semibold">[O] Outcome:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_b.conditions?.outcome_metric || "—"}
                        </span>
                      </div>
                      <div>
                        <span className="text-ink-subtle block font-semibold">[S] Setting:</span>
                        <span className="text-ink font-medium">
                          {selectedPair.finding_b.conditions?.population ||
                            selectedPair.finding_b.conditions?.dataset_setting ||
                            "—"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
