import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error in component:", error, errorInfo);
    this.setState({ error, errorInfo });
    (window as any).__last_react_error = {
      message: error?.message,
      stack: error?.stack,
      componentStack: errorInfo?.componentStack,
    };
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 m-4 rounded-xl border border-destructive/40 bg-destructive/10 text-foreground">
          <div className="flex items-center gap-3 mb-3 text-destructive">
            <AlertTriangle className="w-6 h-6" />
            <h2 className="text-lg font-bold">
              {this.props.fallbackTitle || "Something went wrong in this section"}
            </h2>
          </div>
          <p className="text-sm font-mono text-destructive mb-3">
            {this.state.error?.message || "Unknown error"}
          </p>
          {this.state.error?.stack && (
            <pre className="p-3 bg-black/60 rounded text-xs font-mono text-muted-foreground overflow-auto max-h-60 mb-4 whitespace-pre-wrap">
              {this.state.error.stack}
            </pre>
          )}
          {this.state.errorInfo?.componentStack && (
            <pre className="p-3 bg-black/40 rounded text-xs font-mono text-muted-foreground overflow-auto max-h-40 mb-4 whitespace-pre-wrap">
              {this.state.errorInfo.componentStack}
            </pre>
          )}
          <button
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-xs font-semibold flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
