export type SessionPhase = "open" | "snapshotting" | "restoring" | "closing" | "closed";

export interface TransitionLease {
  generation: number;
  release(): void;
}

/**
 * A synchronous per-session transition gate. It deliberately has no queued
 * mutex: callers must not hold a lock while waiting for prompt cleanup,
 * cancellation, or a mode update.
 */
export class SessionLifecycle {
  phase: SessionPhase = "open";
  generation = 0;
  closeRequested = false;

  begin(phase: "snapshotting" | "restoring" | "closing"): TransitionLease {
    if (this.phase !== "open" || this.closeRequested) {
      throw new Error(`Session transition in progress: ${this.phase}`);
    }
    this.phase = phase;
    if (phase === "closing") this.closeRequested = true;
    const generation = ++this.generation;
    let released = false;
    return {
      generation,
      release: () => {
        if (released) return;
        released = true;
        if (this.generation === generation && this.phase === phase) {
          this.phase = this.closeRequested ? "closing" : "open";
        }
      },
    };
  }

  owns(lease: TransitionLease): boolean {
    return this.generation === lease.generation && this.phase !== "open" && this.phase !== "closed";
  }

  requestClose(): void {
    this.closeRequested = true;
  }

  acceptsPrompts(): boolean {
    return this.phase === "open";
  }
}
