"use client";

import React from "react";
import { AgentEvent } from "../lib/types";

interface TimelineDrawerProps {
  events: AgentEvent[];
  totalDurationMs?: number;
}

export function TimelineDrawer({ events, totalDurationMs }: TimelineDrawerProps) {
  if (!events || events.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-xs text-slate-500 text-center">
        No agent trace events logged.
      </div>
    );
  }

  const agentLabels: Record<string, { name: string; icon: string }> = {
    input_processing_agent: { name: "Agent 1: Input Processing", icon: "📝" },
    evidence_retrieval_agent: { name: "Agent 2: Evidence Retrieval", icon: "🔍" },
    evidence_analysis_agent: { name: "Agent 3: Evidence Analysis & Syndication", icon: "📊" },
    adjudication_agent: { name: "Agent 4: Multi-Source Adjudication", icon: "⚖️" },
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-base">⏱️</span>
          <h4 className="text-sm font-semibold text-white">Agentic Orchestration Trace</h4>
        </div>
        {totalDurationMs !== undefined && (
          <span className="rounded bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-[11px] font-mono text-indigo-300">
            Total: {totalDurationMs} ms
          </span>
        )}
      </div>

      <div className="relative pl-6 space-y-4 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
        {events.map((evt, idx) => {
          const info = agentLabels[evt.agent_name] || { name: evt.agent_name, icon: "⚙️" };
          const isCompleted = evt.status === "completed" || evt.event_type.endsWith("_completed");

          return (
            <div key={idx} className="relative group">
              {/* Dot */}
              <div
                className={`absolute -left-6 top-1 h-3.5 w-3.5 rounded-full border-2 ${
                  isCompleted
                    ? "border-emerald-500 bg-slate-950"
                    : "border-indigo-500 bg-slate-950 animate-pulse"
                }`}
              />

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs">{info.icon}</span>
                  <span className="text-xs font-semibold text-slate-200">{info.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-400 font-mono">
                    {evt.duration_ms} ms
                  </span>
                  <span
                    className={`rounded px-1.5 py-0.2 text-[9px] font-medium uppercase ${
                      isCompleted ? "bg-emerald-500/10 text-emerald-400" : "bg-indigo-500/10 text-indigo-400"
                    }`}
                  >
                    {evt.status}
                  </span>
                </div>
              </div>

              {evt.structured_details && Object.keys(evt.structured_details).length > 0 && (
                <div className="mt-1 text-[11px] text-slate-400 font-mono">
                  {Object.entries(evt.structured_details).map(([k, v]) => (
                    <span key={k} className="mr-3">
                      <span className="text-slate-400">{k}:</span>{" "}
                      <span className="text-slate-200">{String(v)}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
