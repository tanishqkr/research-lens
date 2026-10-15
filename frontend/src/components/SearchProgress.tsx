"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Loader2, CheckCircle2, Search, Filter, BookOpen, GitCompare, Sparkles } from "lucide-react";

interface SearchProgressProps {
  query: string;
}

const STAGES = [
  { id: 1, label: "Expanding research query & identifying keywords", icon: Search, duration: 2000 },
  { id: 2, label: "Querying scholarly providers (OpenAlex & arXiv)", icon: BookOpen, duration: 4000 },
  { id: 3, label: "Screening abstracts & executing empirical triage", icon: Filter, duration: 4500 },
  { id: 4, label: "Extracting verbatim findings & PICO conditions (I, C, O, S)", icon: Sparkles, duration: 5500 },
  { id: 5, label: "Evaluating cross-paper compatibility & condition differences", icon: GitCompare, duration: 4000 },
  { id: 6, label: "Reconciling evidence ledger & synthesizing executive summary", icon: CheckCircle2, duration: 3000 },
];

export function SearchProgress({ query }: SearchProgressProps) {
  const [activeStage, setActiveStage] = useState(1);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let current = 1;
    const timeouts: NodeJS.Timeout[] = [];

    let accumulatedTime = 0;
    STAGES.forEach((stage, idx) => {
      if (idx === 0) return;
      accumulatedTime += STAGES[idx - 1].duration;
      const t = setTimeout(() => {
        setActiveStage(stage.id);
      }, accumulatedTime);
      timeouts.push(t);
    });

    return () => {
      timeouts.forEach(clearTimeout);
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="rounded-xl border border-border bg-surface p-6 shadow-card space-y-5"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-4">
        <div>
          <h3 className="text-base font-semibold text-ink flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-accent animate-spin" />
            Analyzing Scientific Literature
          </h3>
          <p className="text-xs text-ink-muted mt-0.5 line-clamp-1 italic">
            &ldquo;{query}&rdquo;
          </p>
        </div>
        <div className="text-xs font-mono text-ink-subtle">
          Elapsed: {elapsedSeconds}s
        </div>
      </div>

      <div className="space-y-3">
        {STAGES.map((stage) => {
          const isDone = stage.id < activeStage;
          const isCurrent = stage.id === activeStage;
          const Icon = stage.icon;

          return (
            <div
              key={stage.id}
              className={`flex items-center gap-3 p-2.5 rounded-lg text-xs transition-all ${
                isCurrent
                  ? "bg-accent-surface text-accent-text border border-accent-border font-medium"
                  : isDone
                  ? "text-ink-muted"
                  : "text-ink-subtle opacity-60"
              }`}
            >
              <div className="w-5 h-5 flex items-center justify-center shrink-0">
                {isDone ? (
                  <CheckCircle2 className="w-4 h-4 text-evidence-supports-text" />
                ) : isCurrent ? (
                  <Loader2 className="w-4 h-4 text-accent animate-spin" />
                ) : (
                  <Icon className="w-3.5 h-3.5 text-ink-subtle" />
                )}
              </div>
              <span className="flex-1">{stage.label}</span>
              {isDone && (
                <span className="text-[10px] uppercase font-semibold text-evidence-supports-text tracking-wider">
                  Done
                </span>
              )}
              {isCurrent && (
                <span className="text-[10px] uppercase font-semibold text-accent-text tracking-wider animate-pulse">
                  In Progress
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div className="text-[11px] text-ink-subtle text-center pt-1">
        Executing offline-first Python reasoning pipeline with OpenAlex / arXiv retrieval.
      </div>
    </motion.div>
  );
}
