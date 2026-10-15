"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Compass,
  Scale,
  Sparkles,
  Layers,
  ArrowRight,
  BookOpen,
  Split,
  FileCheck2,
} from "lucide-react";

interface EmptyStateProps {
  onSelectQuery: (query: string, autoAnalyze?: boolean) => void;
  presets: string[];
}

export function EmptyState({ onSelectQuery, presets }: EmptyStateProps) {
  const featuredQuestions = [
    {
      title: "Nutrition & Metabolism",
      query:
        "Do randomized controlled trials show that low-carbohydrate diets produce greater weight loss than low-fat diets, and does the difference persist over time?",
      badge: "RCT vs Observational",
      description: "Direct comparison with time-horizon condition differences.",
    },
    {
      title: "Epidemiology & Confounding",
      query:
        "Does moderate alcohol consumption reduce all-cause mortality, or can the apparent protective association be explained by study design and confounding?",
      badge: "Confounding & Bias",
      description: "Sick-quitter reference bias and observational vs mendelian randomization.",
    },
    {
      title: "Preventive Medicine",
      query:
        "Does vitamin D supplementation prevent cardiovascular events in generally healthy adults, and how do randomized trials compare with observational findings?",
      badge: "Null Effect vs Association",
      description: "Large-scale trials (VITAL) vs observational cohort disparities.",
    },
    {
      title: "Clinical Machine Learning",
      query:
        "Which machine-learning approaches have been evaluated for early prediction of pediatric sepsis, and how reliable is the evidence that one approach outperforms another?",
      badge: "Prediction Model",
      description: "Model discrimination, calibration, and external validation conditions.",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-10 py-6 w-full"
    >
      {/* Hero Banner */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05 }}
        className="text-center space-y-4 max-w-4xl mx-auto"
      >
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-semibold bg-accent-surface text-accent-text border border-accent-border">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Condition-Aware Scientific Discovery Engine</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold text-ink tracking-tight">
          Uncover and Reconcile Conflicting Scientific Findings
        </h1>
        <p className="text-sm sm:text-base text-ink-muted leading-relaxed">
          Traditional literature tools classify scientific claims as binary contradictions.
          Our condition-aware workbench disambiguates findings across study designs,
          intervention arms, comparator conditions, and follow-up horizons.
        </p>
      </motion.div>

      {/* Feature Pillars */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-5"
      >
        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 shadow-card hover:border-border-strong transition-all">
          <div className="w-9 h-9 rounded-lg bg-evidence-qualifies-bg text-evidence-qualifies-text border border-evidence-qualifies-border flex items-center justify-center">
            <Layers className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-ink">
            PICO Condition Extraction ($I, C, O, S$)
          </h3>
          <p className="text-xs text-ink-muted leading-relaxed">
            Extracts verbatim passages, intervention dosages, comparator treatments,
            outcome metrics, and patient population constraints.
          </p>
        </div>

        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 shadow-card hover:border-border-strong transition-all">
          <div className="w-9 h-9 rounded-lg bg-evidence-incomparable-bg text-evidence-incomparable-text border border-evidence-incomparable-border flex items-center justify-center">
            <Split className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-ink">
            5-Class Fine-Grained Relations
          </h3>
          <p className="text-xs text-ink-muted leading-relaxed">
            Distinguishes true contradiction from condition incomparability, partial qualification,
            consistent findings, or insufficient empirical evidence.
          </p>
        </div>

        <div className="rounded-xl border border-border bg-surface p-6 space-y-3 shadow-card hover:border-border-strong transition-all">
          <div className="w-9 h-9 rounded-lg bg-evidence-supports-bg text-evidence-supports-text border border-evidence-supports-border flex items-center justify-center">
            <FileCheck2 className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-ink">
            Verbatim Passage Provenance
          </h3>
          <p className="text-xs text-ink-muted leading-relaxed">
            Every finding links directly to exact source passages, DOI citations, and
            structured numerical effect sizes, eliminating hallucinations.
          </p>
        </div>
      </motion.div>

      {/* Recommended Scientific Queries */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-accent" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-ink-muted">
              Select an Empirical Research Question
            </h2>
          </div>
          <span className="text-xs text-ink-subtle">Click to analyze instantly</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 lg:gap-6">
          {featuredQuestions.map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 + idx * 0.05 }}
              whileHover={{ y: -2, transition: { duration: 0.15 } }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSelectQuery(item.query, true)}
              className="text-left rounded-xl border border-border bg-surface p-6 space-y-3.5 shadow-card hover:bg-surface-hover hover:border-border-strong transition-colors group flex flex-col justify-between overflow-hidden cursor-pointer"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-accent-text">
                    {item.title}
                  </span>
                  <span className="text-[11px] font-medium px-2.5 py-0.5 rounded bg-surface-subtle border border-border text-ink-muted">
                    {item.badge}
                  </span>
                </div>
                <p className="text-sm font-medium text-ink group-hover:text-accent transition-colors line-clamp-3 leading-relaxed">
                  &ldquo;{item.query}&rdquo;
                </p>
                <div className="text-xs text-ink-subtle leading-normal">
                  {item.description}
                </div>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-border/50 text-xs">
                <motion.button
                  type="button"
                  whileHover={{ y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectQuery(item.query, false);
                  }}
                  className="text-ink-muted hover:text-ink font-medium cursor-pointer py-1"
                >
                  Edit Question
                </motion.button>
                <motion.button
                  type="button"
                  whileHover={{ y: -1 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectQuery(item.query, true);
                  }}
                  className="inline-flex items-center gap-1.5 font-semibold text-accent hover:text-accent-hover cursor-pointer py-1"
                >
                  <span>Analyze Now</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Additional Curated Presets Pills */}
      {presets.length > 4 && (
        <div className="space-y-3 pt-2">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-subtle flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5" />
            More Curated Presets
          </div>
          <div className="flex flex-wrap gap-2">
            {presets.slice(4).map((p, idx) => (
              <motion.button
                key={idx}
                whileHover={{ y: -1 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onSelectQuery(p)}
                className="text-xs px-3.5 py-2 rounded-lg border border-border bg-surface hover:bg-surface-hover text-ink-muted hover:text-ink transition-colors cursor-pointer text-left line-clamp-1"
              >
                {p}
              </motion.button>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
