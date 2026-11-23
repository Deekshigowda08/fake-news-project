"use client";

import React from "react";
import { SystemHealth } from "../lib/types";

interface NavbarProps {
  activeTab: "analyze" | "results" | "history" | "research" | "system";
  setActiveTab: (tab: "analyze" | "results" | "history" | "research" | "system") => void;
  systemHealth: SystemHealth | null;
  hasActiveResult: boolean;
}

export function Navbar({ activeTab, setActiveTab, systemHealth, hasActiveResult }: NavbarProps) {
  const isHealthy = systemHealth?.status === "healthy";

  return (
    <header className="sticky top-0 z-50 border-b border-zinc-800 bg-black/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        {/* Brand / Logo */}
        <div 
          className="flex items-center space-x-3 cursor-pointer group" 
          onClick={() => setActiveTab("analyze")}
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-white text-black font-black transition-all">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight text-white">
                Fake News Detector
              </span>
              <span className="hidden sm:inline-block rounded border border-zinc-700 bg-zinc-900 px-2 py-0.5 text-[10px] font-mono font-medium text-zinc-300 uppercase">
                CNN+LSTM & Agents
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">Deep Learning & Web Evidence Verification</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 bg-zinc-900 p-1 rounded-xl border border-zinc-800">
          <button
            id="nav-tab-analyze"
            onClick={() => setActiveTab("analyze")}
            className={`rounded-lg px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-colors ${
              activeTab === "analyze"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-zinc-400 hover:text-white hover:bg-zinc-800"
            }`}
          >
            Analyze News
          </button>

          <button
            id="nav-tab-results"
            onClick={() => setActiveTab("results")}
            disabled={!hasActiveResult}
            className={`rounded-lg px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-colors ${
              !hasActiveResult
                ? "cursor-not-allowed text-zinc-600"
                : activeTab === "results"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-zinc-400 hover:text-white hover:bg-zinc-800"
            }`}
          >
            Results
          </button>

          <button
            id="nav-tab-history"
            onClick={() => setActiveTab("history")}
            className={`rounded-lg px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-colors ${
              activeTab === "history"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-zinc-400 hover:text-white hover:bg-zinc-800"
            }`}
          >
            History
          </button>

          <button
            id="nav-tab-research"
            onClick={() => setActiveTab("research")}
            className={`rounded-lg px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-colors ${
              activeTab === "research"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-zinc-400 hover:text-white hover:bg-zinc-800"
            }`}
          >
            Architecture
          </button>
        </nav>

        {/* Live Server Indicator */}
        <div className="hidden lg:flex items-center space-x-2 rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-1.5 text-xs">
          <span className={`h-2 w-2 rounded-full ${isHealthy ? "bg-white" : "bg-zinc-600"}`} />
          <span className="font-mono text-zinc-300">
            {isHealthy ? `Backend: ${systemHealth?.device?.toUpperCase() || "ONLINE"}` : "Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
