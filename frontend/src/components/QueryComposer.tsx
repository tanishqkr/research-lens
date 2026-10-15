"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "./ui/Button";
import { Search, SlidersHorizontal, Sparkles, BookOpen, Layers } from "lucide-react";

interface QueryComposerProps {
  query: string;
  setQuery: (q: string) => void;
  onAnalyze: (options: { limit: number; useGroq: boolean }) => void;
  isLoading: boolean;
  presets: string[];
}

export function QueryComposer({
  query,
  setQuery,
  onAnalyze,
  isLoading,
  presets,
}: QueryComposerProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [limit, setLimit] = useState(10);
  const [useGroq, setUseGroq] = useState(false);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;
    onAnalyze({ limit, useGroq });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePresetSelect = (preset: string) => {
    setQuery(preset);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="w-full rounded-xl border border-border bg-surface p-5 sm:p-6 shadow-card transition-all"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Preset selector */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <label
            htmlFor="query-input"
            className="text-xs font-semibold uppercase tracking-wider text-ink-muted flex items-center gap-1.5"
          >
            <BookOpen className="w-3.5 h-3.5 text-accent" />
            Research Question or Empirical Claim
          </label>

          <div className="flex items-center gap-2 w-full sm:w-auto">
            <span className="text-xs text-ink-subtle whitespace-nowrap hidden md:inline">
              Examples:
            </span>
            <select
              aria-label="Select empirical query preset"
              className="text-xs rounded border border-border bg-surface-subtle px-2.5 py-1 text-ink focus:outline-none focus:ring-1 focus:ring-accent max-w-full sm:max-w-[320px] truncate cursor-pointer"
              onChange={(e) => {
                if (e.target.value) handlePresetSelect(e.target.value);
              }}
              defaultValue=""
            >
              <option value="" disabled>
                Select Curated Scientific Preset...
              </option>
              {presets.map((p, idx) => (
                <option key={idx} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Query Input Area */}
        <div className="relative rounded-lg border border-border bg-surface-subtle focus-within:border-accent-border focus-within:ring-2 focus-within:ring-accent/20 transition-all">
          <textarea
            id="query-input"
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. Do randomized controlled trials show that low-carbohydrate diets produce greater weight loss than low-fat diets, and does the difference persist over time?"
            className="w-full bg-transparent px-4 py-3 text-sm text-ink placeholder:text-ink-subtle focus:outline-none resize-none"
            disabled={isLoading}
          />

          <div className="flex items-center justify-between px-3 py-2 border-t border-border/60 bg-surface/40 text-xs text-ink-subtle">
            <span className="hidden sm:inline">
              Press <kbd className="px-1.5 py-0.5 rounded bg-surface-subtle border border-border text-[10px] font-mono">⌘+Enter</kbd> to analyze
            </span>
            <span className="sm:hidden text-[11px]">
              Tap Analyze to retrieve literature
            </span>

            <motion.button
              type="button"
              whileTap={{ scale: 0.98 }}
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-ink-muted hover:text-ink font-medium transition-colors cursor-pointer"
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              <span>{showAdvanced ? "Hide Controls" : "Advanced Controls"}</span>
            </motion.button>
          </div>
        </div>

        {/* Advanced Controls Panel */}
        {showAdvanced && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className="rounded-lg border border-border-subtle bg-surface-subtle p-4 text-xs space-y-3"
          >
            <div className="font-semibold text-ink flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-accent" />
              Pipeline Configuration
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-ink-muted mb-1 font-medium">
                  Candidate Papers Retrieval Limit: <span className="text-ink font-mono font-semibold">{limit}</span>
                </label>
                <input
                  type="range"
                  min={5}
                  max={25}
                  step={5}
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  className="w-full accent-accent cursor-pointer"
                  disabled={isLoading}
                />
                <div className="flex justify-between text-[10px] text-ink-subtle mt-0.5">
                  <span>5 (Fast triage)</span>
                  <span>15 (Balanced)</span>
                  <span>25 (Deep coverage)</span>
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-ink-muted font-medium">
                  Analysis Pipeline Features:
                </label>
                <label className="flex items-center gap-2 text-ink cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useGroq}
                    onChange={(e) => setUseGroq(e.target.checked)}
                    disabled={isLoading}
                    className="rounded border-border text-accent focus:ring-accent accent-accent"
                  />
                  <span>Enable LLM Candidate Sentence Auditor</span>
                </label>
                <p className="text-[11px] text-ink-subtle pl-5">
                  Audits extracted candidate sentences against source abstracts when API credentials are provided.
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-border flex items-center justify-between text-[11px] text-ink-subtle">
              <span className="flex items-center gap-1">
                <span>Providers queried:</span>
                <span className="font-medium text-ink">OpenAlex (primary)</span>
                <span>•</span>
                <span className="font-medium text-ink">arXiv (preprints)</span>
              </span>
              <span>Condition extraction: <strong>PICO ($I, C, O, S$)</strong></span>
            </div>
          </motion.div>
        )}

        {/* Submit action button */}
        <div className="flex justify-end pt-1">
          <Button
            type="submit"
            variant="primary"
            size="md"
            isLoading={isLoading}
            disabled={!query.trim()}
            className="w-full sm:w-auto px-6"
          >
            <Search className="w-4 h-4" />
            Analyze Empirical Conflict
          </Button>
        </div>
      </form>
    </motion.div>
  );
}
