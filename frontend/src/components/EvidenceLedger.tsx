"use client";

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  SupportedEvidenceClaim,
  Finding,
  DirectionLabel,
  StudyDesignType,
  NumericalResult,
} from "@/types/scientific";
import { Badge, DirectionBadge, StudyDesignBadge } from "./ui/Badge";
import {
  BookOpen,
  Filter,
  Search,
  ExternalLink,
  Quote,
  Layers,
  Activity,
  AlertCircle,
  FileText,
  Clock,
  Users,
  ChevronDown,
  ChevronUp,
  Cpu,
} from "lucide-react";

interface EvidenceLedgerProps {
  ledger?: SupportedEvidenceClaim[];
  findings?: Finding[];
}

// Unified view model for presentation
interface NormalizedClaim {
  id: string;
  paperId: string;
  title: string;
  doi?: string | null;
  url?: string | null;
  passage: string;
  studyDesign?: string | null;
  studyType: StudyDesignType;
  provenanceSection: string;
  direction: DirectionLabel;
  fullTextUsed: boolean;
  isDirect: boolean;
  // Slots
  intervention?: string | null;
  comparator?: string | null;
  outcome?: string | null;
  populationSetting?: string | null;
  effectSize?: string | null;
  followUpDuration?: string | null;
  sampleSize?: number | null;
  numericalResults: NumericalResult[];
  limitations?: string | null;
  predictionModelDetails?: any;
}

export function EvidenceLedger({ ledger = [], findings = [] }: EvidenceLedgerProps) {
  // Normalize items from either ledger (primary) or raw findings (fallback)
  const items: NormalizedClaim[] = useMemo(() => {
    if (ledger && ledger.length > 0) {
      return ledger.map((item) => ({
        id: item.claim_id,
        paperId: item.paper_id,
        title: item.paper_title,
        doi: item.doi,
        url: item.url,
        passage: item.exact_source_passage,
        studyDesign: item.study_design,
        studyType: item.study_type,
        provenanceSection: item.provenance_section,
        direction: item.direction,
        fullTextUsed: item.full_text_used,
        isDirect: item.is_direct,
        intervention: item.intervention_exposure,
        comparator: item.comparator,
        outcome: item.outcome,
        populationSetting: item.population_setting,
        effectSize: item.effect_size,
        followUpDuration: item.follow_up_duration,
        sampleSize: item.sample_size,
        numericalResults: item.numerical_results || [],
        limitations: item.limitations,
        predictionModelDetails: item.prediction_model_details,
      }));
    }

    if (findings && findings.length > 0) {
      return findings.map((f) => ({
        id: f.finding_id,
        paperId: f.paper_id,
        title: f.paper_title,
        doi: f.paper_doi,
        url: f.paper_url,
        passage: f.evidence?.text || f.finding_claim,
        studyDesign: f.conditions?.study_design,
        studyType: f.conditions?.study_type || "UNKNOWN",
        provenanceSection: f.conditions?.provenance_section || "RESULTS",
        direction: f.direction,
        fullTextUsed: false,
        isDirect: true,
        intervention: f.conditions?.intervention,
        comparator: f.conditions?.comparator,
        outcome: f.conditions?.outcome_metric,
        populationSetting: f.conditions?.population || f.conditions?.dataset_setting,
        effectSize: f.conditions?.effect_size,
        followUpDuration: f.conditions?.follow_up_duration,
        sampleSize: f.conditions?.sample_size,
        numericalResults: f.conditions?.numerical_results || [],
        limitations: f.conditions?.limitations_or_qualifiers,
        predictionModelDetails: null,
      }));
    }

    return [];
  }, [ledger, findings]);

  // Filters
  const [directionFilter, setDirectionFilter] = useState<string>("ALL");
  const [designFilter, setDesignFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedDetails, setExpandedDetails] = useState<Record<string, boolean>>({});

  const toggleExpand = (id: string) => {
    setExpandedDetails((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      // Direction filter
      if (directionFilter !== "ALL" && item.direction !== directionFilter) {
        return false;
      }
      // Design filter
      if (designFilter !== "ALL" && item.studyType !== designFilter) {
        return false;
      }
      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesTitle = (item.title || "").toLowerCase().includes(q);
        const matchesPassage = (item.passage || "").toLowerCase().includes(q);
        const matchesIntervention = (item.intervention || "").toLowerCase().includes(q);
        const matchesOutcome = (item.outcome || "").toLowerCase().includes(q);
        if (!matchesTitle && !matchesPassage && !matchesIntervention && !matchesOutcome) {
          return false;
        }
      }
      return true;
    });
  }, [items, directionFilter, designFilter, searchQuery]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-5 w-full"
    >
      {/* Control / Filter Bar */}
      <div className="rounded-xl border border-border bg-surface p-4 sm:p-5 shadow-card space-y-4">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-accent" />
            <h3 className="text-base font-bold text-ink">
              Evidence Ledger & Verbatim Finding Cards
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-surface-subtle border border-border text-ink-muted font-mono font-medium">
              {filteredItems.length} of {items.length} claims
            </span>
          </div>

          {/* Quick text filter */}
          <div className="relative w-full md:w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-ink-subtle" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search claims, passages, metrics..."
              className="w-full text-xs rounded-lg border border-border bg-surface-subtle pl-9 pr-3 py-2 text-ink placeholder:text-ink-subtle focus:outline-none focus:ring-1 focus:ring-accent"
            />
          </div>
        </div>

        {/* Filter Pills */}
        <div className="flex flex-wrap items-center gap-3 pt-1 border-t border-border/70 text-xs">
          <div className="flex items-center gap-1.5 text-ink-muted">
            <Filter className="w-3.5 h-3.5 text-accent" />
            <span className="font-semibold uppercase tracking-wider text-[11px]">
              Direction:
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {[
              { id: "ALL", label: "All Directions" },
              { id: "POSITIVE", label: "Positive (+)" },
              { id: "NEGATIVE", label: "Negative (−)" },
              { id: "NO_DIFFERENCE", label: "No Difference" },
              { id: "MIXED", label: "Mixed Effects" },
              { id: "UNCLEAR", label: "Unclear" },
            ].map((d) => (
              <motion.button
                key={d.id}
                whileHover={{ y: -1 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => setDirectionFilter(d.id)}
                className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
                  directionFilter === d.id
                    ? "bg-accent text-white font-semibold shadow-sm"
                    : "bg-surface-subtle text-ink-muted hover:text-ink hover:bg-surface-hover border border-border"
                }`}
              >
                {d.label}
              </motion.button>
            ))}
          </div>

          <div className="hidden sm:inline text-border">|</div>

          <div className="flex items-center gap-1.5 text-ink-muted">
            <span className="font-semibold uppercase tracking-wider text-[11px]">
              Design:
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {[
              { id: "ALL", label: "All Designs" },
              { id: "INTERVENTIONAL_TRIAL", label: "RCT / Interventional" },
              { id: "OBSERVATIONAL_STUDY", label: "Observational" },
              { id: "SYSTEMATIC_REVIEW_META_ANALYSIS", label: "Systematic Review" },
              { id: "PREDICTION_MODEL", label: "Prediction Model" },
            ].map((t) => (
              <motion.button
                key={t.id}
                whileHover={{ y: -1 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => setDesignFilter(t.id)}
                className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
                  designFilter === t.id
                    ? "bg-accent text-white font-semibold shadow-sm"
                    : "bg-surface-subtle text-ink-muted hover:text-ink hover:bg-surface-hover border border-border"
                }`}
              >
                {t.label}
              </motion.button>
            ))}
          </div>
        </div>
      </div>

      {/* Cards List */}
      {items.length === 0 ? (
        <div className="rounded-xl border border-border bg-surface p-8 text-center text-ink-muted text-sm space-y-2">
          <AlertCircle className="w-6 h-6 mx-auto text-ink-subtle" />
          <p className="font-medium text-ink">No empirical evidence claims extracted for this query.</p>
          <p className="text-xs text-ink-muted">
            The retrieved literature may consist of non-empirical background or lacks explicit quantitative findings.
          </p>
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="rounded-xl border border-border bg-surface p-8 text-center text-ink-muted text-sm space-y-2">
          <AlertCircle className="w-6 h-6 mx-auto text-ink-subtle" />
          <p>No evidence claims match your active filter criteria.</p>
          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              setDirectionFilter("ALL");
              setDesignFilter("ALL");
              setSearchQuery("");
            }}
            className="text-xs text-accent hover:underline font-medium cursor-pointer"
          >
            Reset all filters
          </motion.button>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredItems.map((item) => {
            const isExpanded = !!expandedDetails[item.id];
            const hasMissingComparator = !item.comparator || item.comparator.trim() === "";
            const hasMissingSetting = !item.populationSetting || item.populationSetting.trim() === "";
            const hasMissingOutcome = !item.outcome || item.outcome.trim() === "";

            return (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25 }}
                whileHover={{ y: -2, transition: { duration: 0.15 } }}
                className="rounded-xl border border-border bg-surface shadow-card hover:border-border-strong transition-all overflow-hidden w-full"
              >
                {/* Header row */}
                <div className="p-4 sm:p-5 border-b border-border bg-surface-subtle/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="space-y-1 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <DirectionBadge direction={item.direction} />
                      <StudyDesignBadge design={item.studyType} />
                      <Badge variant="outline" size="sm">
                        SEC: {item.provenanceSection}
                      </Badge>
                      {item.fullTextUsed ? (
                        <Badge variant="supports" size="sm">
                          Full-Text Grounded
                        </Badge>
                      ) : (
                        <Badge variant="default" size="sm">
                          Abstract Only
                        </Badge>
                      )}
                      {item.isDirect && (
                        <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-accent-surface text-accent-text border border-accent-border">
                          Direct
                        </span>
                      )}
                    </div>

                    <h4 className="text-sm sm:text-base font-bold text-ink leading-snug pt-1">
                      {item.title}
                    </h4>
                  </div>

                  {/* External links */}
                  <div className="flex items-center gap-2 shrink-0">
                    {item.doi && (
                      <a
                        href={item.doi.startsWith("http") ? item.doi : `https://doi.org/${item.doi}`}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-accent hover:text-accent-hover font-medium px-2 py-1 rounded border border-border bg-surface hover:bg-surface-hover transition-colors"
                      >
                        <span>DOI</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                    {item.url && !item.doi && (
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-accent hover:text-accent-hover font-medium px-2 py-1 rounded border border-border bg-surface hover:bg-surface-hover transition-colors"
                      >
                        <span>Source</span>
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>
                </div>

                {/* Verbatim Source Passage */}
                <div className="p-4 sm:p-5 space-y-4">
                  <div className="rounded-lg border-l-4 border-accent bg-surface-subtle p-3.5 space-y-1.5">
                    <div className="flex items-center justify-between text-xs text-ink-muted">
                      <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center gap-1 text-accent">
                        <Quote className="w-3 h-3" />
                        Verbatim Source Passage
                      </span>
                      <span className="text-[10px] text-ink-subtle font-mono">
                        {item.paperId}
                      </span>
                    </div>
                    <blockquote className="text-xs sm:text-sm text-ink italic leading-relaxed">
                      &ldquo;{item.passage}&rdquo;
                    </blockquote>
                  </div>

                  {/* PICO Slot Grid (I, C, O, S) */}
                  <div className="space-y-1.5">
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-subtle flex items-center gap-1">
                      <Layers className="w-3 h-3 text-accent" />
                      Extracted Condition Slots (PICO)
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 text-xs">
                      {/* Intervention Slot */}
                      <div className="p-2.5 rounded-lg border border-border bg-surface-subtle space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-accent-text text-[11px]">
                            [I] Intervention / Exposure
                          </span>
                        </div>
                        <p className="text-ink font-medium leading-snug">
                          {item.intervention || (
                            <span className="text-ink-subtle italic">Unspecified</span>
                          )}
                        </p>
                      </div>

                      {/* Comparator Slot */}
                      <div className="p-2.5 rounded-lg border border-border bg-surface-subtle space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-evidence-qualifies-text text-[11px]">
                            [C] Comparator
                          </span>
                          {hasMissingComparator && (
                            <span className="text-[10px] text-evidence-qualifies-text font-medium">
                              Missing
                            </span>
                          )}
                        </div>
                        <p className="text-ink font-medium leading-snug">
                          {item.comparator ? (
                            item.comparator
                          ) : (
                            <span className="text-evidence-qualifies-text/80 italic text-[11px]">
                              No explicit comparator arm
                            </span>
                          )}
                        </p>
                      </div>

                      {/* Outcome Slot */}
                      <div className="p-2.5 rounded-lg border border-border bg-surface-subtle space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-evidence-supports-text text-[11px]">
                            [O] Outcome / Metric
                          </span>
                          {hasMissingOutcome && (
                            <span className="text-[10px] text-evidence-qualifies-text font-medium">
                              Missing
                            </span>
                          )}
                        </div>
                        <p className="text-ink font-medium leading-snug">
                          {item.outcome || (
                            <span className="text-ink-subtle italic">Unspecified</span>
                          )}
                        </p>
                      </div>

                      {/* Setting / Population Slot */}
                      <div className="p-2.5 rounded-lg border border-border bg-surface-subtle space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-evidence-incomparable-text text-[11px]">
                            [S] Population / Setting
                          </span>
                          {hasMissingSetting && (
                            <span className="text-[10px] text-evidence-qualifies-text font-medium">
                              Missing
                            </span>
                          )}
                        </div>
                        <p className="text-ink font-medium leading-snug">
                          {item.populationSetting ? (
                            item.populationSetting
                          ) : (
                            <span className="text-evidence-qualifies-text/80 italic text-[11px]">
                              General / unstated population
                            </span>
                          )}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Quantitative Metrics Bar */}
                  <div className="flex flex-wrap items-center gap-3 pt-1 text-xs">
                    {item.sampleSize && (
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-subtle border border-border text-ink">
                        <Users className="w-3.5 h-3.5 text-accent" />
                        <span>Sample Size ($N$):</span>
                        <strong className="font-mono">{item.sampleSize.toLocaleString()}</strong>
                      </div>
                    )}

                    {item.followUpDuration && (
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-subtle border border-border text-ink">
                        <Clock className="w-3.5 h-3.5 text-accent" />
                        <span>Horizon:</span>
                        <strong>{item.followUpDuration}</strong>
                      </div>
                    )}

                    {item.effectSize && (
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-subtle border border-border text-ink">
                        <Activity className="w-3.5 h-3.5 text-evidence-supports-text" />
                        <span>Reported Effect:</span>
                        <strong className="font-mono">{item.effectSize}</strong>
                      </div>
                    )}
                  </div>

                  {/* Numerical Results Table (if present) */}
                  {item.numericalResults && item.numericalResults.length > 0 && (
                    <div className="space-y-1.5 pt-1">
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-subtle flex items-center gap-1">
                        <Activity className="w-3 h-3 text-accent" />
                        Granular Numerical Results ({item.numericalResults.length})
                      </div>
                      <div className="overflow-x-auto rounded-lg border border-border">
                        <table className="w-full text-xs text-left">
                          <thead className="bg-surface-subtle text-ink-muted border-b border-border text-[11px] uppercase font-semibold">
                            <tr>
                              <th className="px-3 py-2">Arm / Metric</th>
                              <th className="px-3 py-2">Point Estimate</th>
                              <th className="px-3 py-2">95% CI</th>
                              <th className="px-3 py-2">p-value</th>
                              <th className="px-3 py-2">Follow-up / N</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border bg-surface font-mono">
                            {item.numericalResults.map((nr, idx) => (
                              <tr key={idx} className="hover:bg-surface-hover/50">
                                <td className="px-3 py-1.5 font-sans font-medium text-ink">
                                  {nr.arm_name || nr.endpoint || nr.metric_name || "—"}
                                  {nr.comparator_arm && (
                                    <span className="text-ink-muted font-normal block text-[11px]">
                                      vs {nr.comparator_arm}
                                    </span>
                                  )}
                                </td>
                                <td className="px-3 py-1.5 text-ink font-semibold">
                                  {nr.point_estimate !== null && nr.point_estimate !== undefined
                                    ? `${nr.point_estimate} ${nr.unit || ""}`
                                    : nr.raw_text || "—"}
                                </td>
                                <td className="px-3 py-1.5 text-ink-muted">
                                  {nr.ci_lower !== null && nr.ci_upper !== null && nr.ci_lower !== undefined
                                    ? `[${nr.ci_lower}, ${nr.ci_upper}]`
                                    : "—"}
                                </td>
                                <td className="px-3 py-1.5 text-ink-muted">
                                  {nr.p_value !== null && nr.p_value !== undefined
                                    ? `p = ${nr.p_value}`
                                    : "—"}
                                </td>
                                <td className="px-3 py-1.5 text-ink-subtle font-sans">
                                  {nr.follow_up_horizon || (nr.sample_size ? `N=${nr.sample_size}` : "—")}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Expandable Limitations / Prediction Model details */}
                  {(item.limitations || item.predictionModelDetails) && (
                    <div className="pt-1">
                      <motion.button
                        whileTap={{ scale: 0.98 }}
                        onClick={() => toggleExpand(item.id)}
                        className="flex items-center gap-1.5 text-xs text-accent hover:text-accent-hover font-medium cursor-pointer"
                      >
                        {isExpanded ? (
                          <>
                            <ChevronUp className="w-3.5 h-3.5" />
                            <span>Hide Study Limitations & Model Architecture</span>
                          </>
                        ) : (
                          <>
                            <ChevronDown className="w-3.5 h-3.5" />
                            <span>View Study Limitations & Model Details</span>
                          </>
                        )}
                      </motion.button>

                      {isExpanded && (
                        <div className="mt-2.5 p-3 rounded-lg border border-border bg-surface-subtle text-xs space-y-2">
                          {item.limitations && (
                            <div>
                              <strong className="text-ink font-semibold block text-[11px] uppercase tracking-wider">
                                Reported Limitations / Qualifiers:
                              </strong>
                              <p className="text-ink-muted mt-0.5 leading-relaxed">
                                {item.limitations}
                              </p>
                            </div>
                          )}

                          {item.predictionModelDetails && (
                            <div className="pt-2 border-t border-border space-y-1">
                              <div className="flex items-center gap-1 font-semibold text-ink text-[11px] uppercase tracking-wider">
                                <Cpu className="w-3.5 h-3.5 text-accent" />
                                Prediction Model Details
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-ink-muted font-mono text-[11px]">
                                {item.predictionModelDetails.model_method && (
                                  <div>
                                    Method: <strong className="text-ink">{item.predictionModelDetails.model_method}</strong>
                                  </div>
                                )}
                                {item.predictionModelDetails.validation_design && (
                                  <div>
                                    Validation: <strong className="text-ink">{item.predictionModelDetails.validation_design}</strong>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
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
