"use client";

import { useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/Button";

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Workbench Client Error Boundary Caught:", error);
  }, [error]);

  return (
    <div className="min-h-[400px] flex flex-col items-center justify-center p-8 text-center space-y-4">
      <div className="p-3 rounded-full bg-evidence-contradicts-bg border border-evidence-contradicts-border text-evidence-contradicts-text">
        <AlertCircle className="w-8 h-8" />
      </div>
      <div className="space-y-1 max-w-md">
        <h2 className="text-lg font-bold text-ink">Application Component Error</h2>
        <p className="text-xs text-ink-muted leading-relaxed">
          {error.message || "An unexpected error occurred while rendering the scientific analysis deck."}
        </p>
      </div>
      <Button onClick={() => reset()} variant="secondary" size="sm">
        <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
        Reload Workbench View
      </Button>
    </div>
  );
}
