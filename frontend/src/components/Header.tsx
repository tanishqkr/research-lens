"use client";

import React from "react";
import { motion } from "framer-motion";
import { useTheme } from "@/context/ThemeContext";
import { Moon, Sun, Microscope, GitCompare, Search, Database } from "lucide-react";

interface HeaderProps {
  activeTab: "search" | "benchmark";
  setActiveTab: (tab: "search" | "benchmark") => void;
  isBackendConnected: boolean | null;
  offlineMode?: boolean;
}

export function Header({
  activeTab,
  setActiveTab,
  isBackendConnected,
  offlineMode,
}: HeaderProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <motion.header
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="sticky top-0 z-30 border-b border-border bg-surface/90 backdrop-blur-md transition-colors"
    >
      <div className="w-full max-w-[1440px] px-6 md:px-12 lg:px-16 mx-auto h-16 flex items-center justify-between gap-4">
        {/* Brand identity */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-md bg-accent text-white flex items-center justify-center shadow-sm">
            <Microscope className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-base tracking-tight text-ink">
                Scientific Contradiction Finder
              </span>
              <span className="hidden sm:inline-block text-[11px] font-medium tracking-wide uppercase px-2 py-0.5 rounded bg-accent-surface text-accent-text border border-accent-border">
                Condition-Aware
              </span>
            </div>
            <p className="text-xs text-ink-muted hidden md:block">
              Condition-aware discovery and analysis of conflicting scientific findings
            </p>
          </div>
        </div>

        {/* Center navigation tabs */}
        <nav className="flex items-center p-1 rounded-md bg-surface-subtle border border-border text-sm">
          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab("search")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-medium transition-colors cursor-pointer ${
              activeTab === "search"
                ? "bg-surface text-ink shadow-sm font-semibold"
                : "text-ink-muted hover:text-ink"
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Live Empirical Search</span>
            <span className="sm:hidden">Search</span>
          </motion.button>
          <motion.button
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab("benchmark")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-medium transition-colors cursor-pointer ${
              activeTab === "benchmark"
                ? "bg-surface text-ink shadow-sm font-semibold"
                : "text-ink-muted hover:text-ink"
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Benchmark Explorer</span>
            <span className="sm:hidden">CAS-Con150</span>
          </motion.button>
        </nav>

        {/* Right tools: Status and Theme toggle */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* System status pill */}
          <div
            className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded text-xs border border-border bg-surface-subtle"
            title={
              isBackendConnected === true
                ? offlineMode
                  ? "Backend connected (Offline Cache Mode: $0 spend)"
                  : "Backend connected (Live Scholarly APIs enabled)"
                : isBackendConnected === false
                ? "Backend not reachable on port 8000"
                : "Checking backend status..."
            }
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isBackendConnected === true
                  ? "bg-emerald-500"
                  : isBackendConnected === false
                  ? "bg-rose-500"
                  : "bg-amber-500 animate-pulse"
              }`}
            />
            <span className="text-ink-muted">
              {isBackendConnected === true
                ? offlineMode
                  ? "Offline ($0 API)"
                  : "Live API"
                : isBackendConnected === false
                ? "API Disconnected"
                : "Connecting..."}
            </span>
          </div>

          {/* Theme Switcher Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
            className="p-2 rounded-md border border-border bg-surface-subtle hover:bg-surface-hover text-ink-muted hover:text-ink transition-colors cursor-pointer"
          >
            {theme === "light" ? (
              <Moon className="w-4 h-4 text-ink-muted" />
            ) : (
              <Sun className="w-4 h-4 text-amber-400" />
            )}
          </motion.button>
        </div>
      </div>
    </motion.header>
  );
}
