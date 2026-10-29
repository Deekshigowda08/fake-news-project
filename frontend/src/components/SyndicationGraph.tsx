"use client";

import React from "react";

interface SyndicationGraphProps {
  claimAnalyses?: any[];
}

export function SyndicationGraph({ claimAnalyses }: SyndicationGraphProps) {
  if (!claimAnalyses || claimAnalyses.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base">🕸️</span>
          <div>
            <h4 className="text-sm font-semibold text-white">Source Independence & Syndication Audit</h4>
            <p className="text-[11px] text-slate-400">
              Clustering wire service syndications vs independent investigative journalism
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {claimAnalyses.map((ca, idx) => {
          const clusters = ca.syndication_clusters || [];
          const indepCount = ca.independent_sources_count || 0;
          const totalCount = ca.total_sources_evaluated || 0;

          return (
            <div key={idx} className="rounded-lg border border-slate-800/80 bg-slate-950/40 p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300">
                  Claim {idx + 1}: "{ca.claim_id}"
                </span>
                <span className="rounded bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-[10px] font-mono text-indigo-300">
                  {indepCount} Independent Clusters / {totalCount} Sources
                </span>
              </div>

              {clusters.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
                  {clusters.map((cluster: any, cIdx: number) => (
                    <div
                      key={cIdx}
                      className="rounded border border-slate-800 bg-slate-900/80 p-2 text-xs space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-200">
                          {cluster.cluster_label || `Cluster ${cIdx + 1}`}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          {cluster.sources?.length || 1} articles
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-400">
                        Root: <span className="text-indigo-400">{cluster.root_publisher || "Unknown"}</span>
                      </div>
                      {cluster.is_wire_syndication && (
                        <span className="inline-block rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.2 text-[9px]">
                          Wire Service Syndication (Single Root Story)
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-400">
                  No wire duplicates identified; evaluating as distinct reporting entities.
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
