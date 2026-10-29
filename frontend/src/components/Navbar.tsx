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
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 backdrop-blur-md shadow-lg shadow-black/20">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3.5 sm:px-6">
        {/* Brand */}
        <div className="flex items-center space-x-3.5 cursor-pointer" onClick={() => setActiveTab("analyze")}>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-lg sm:text-xl font-black tracking-tight text-white">
                Fake News Detection & Agentic Verification
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1.5">
          <button
            id="nav-tab-analyze"
            onClick={() => setActiveTab("analyze")}
            className={`rounded-xl px-4 py-2.5 text-sm font-bold transition-all ${
              activeTab === "analyze"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                : "text-slate-200 hover:bg-slate-800/80 hover:text-white"
            }`}
          >
            Analyze News
          </button>

          <button
            id="nav-tab-results"
            onClick={() => setActiveTab("results")}
            disabled={!hasActiveResult}
            className={`rounded-xl px-4 py-2.5 text-sm font-bold transition-all ${
              !hasActiveResult
                ? "cursor-not-allowed text-slate-600"
                : activeTab === "results"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                : "text-slate-200 hover:bg-slate-800/80 hover:text-white"
            }`}
          >
            Inspection & Results
          </button>

          <button
            id="nav-tab-history"
            onClick={() => setActiveTab("history")}
            className={`rounded-xl px-4 py-2.5 text-sm font-bold transition-all ${
              activeTab === "history"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                : "text-slate-200 hover:bg-slate-800/80 hover:text-white"
            }`}
          >
            History
          </button>

          <button
            id="nav-tab-research"
            onClick={() => setActiveTab("research")}
            className={`hidden md:inline-flex rounded-xl px-4 py-2.5 text-sm font-bold transition-all ${
              activeTab === "research"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                : "text-slate-200 hover:bg-slate-800/80 hover:text-white"
            }`}
          >
            Research Matrix
          </button>

          <button
            id="nav-tab-system"
            onClick={() => setActiveTab("system")}
            className={`hidden lg:inline-flex rounded-xl px-4 py-2.5 text-sm font-bold transition-all ${
              activeTab === "system"
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/40"
                : "text-slate-200 hover:bg-slate-800/80 hover:text-white"
            }`}
          >
            System Status
          </button>
        </nav>

        {/* Backend & GPU Status Badge */}
        <div className="flex items-center space-x-2">
          <div
            className={`flex items-center space-x-2 rounded-full px-3 py-1.5 text-xs font-semibold border ${
              isHealthy
                ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                : "border-amber-500/40 bg-amber-500/15 text-amber-300"
            }`}
          >
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                isHealthy ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            <span>{isHealthy ? "API Online" : "Checking API..."}</span>
          </div>

          {systemHealth?.gpu_available && (
            <div className="hidden sm:flex items-center space-x-1.5 rounded-full px-3 py-1.5 text-xs font-bold border border-cyan-500/40 bg-cyan-500/15 text-cyan-300">
              <span>⚡ CUDA Active</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
