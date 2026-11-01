import type { AdminSettings } from "./api";

// What survives when fresh server values reach the form: "all" typed values, or only the "service"-wide ones.
export type KeepEdits = "all" | "service";

// Puts what the admin typed on top of fresh server values. A field counts as typed when it differs from what the
// server sent last, so everything else follows the server. Without keepEdits the server values replace the form.
// Typed model settings belong to the model the form was loaded for: when another model is active now, they would pass
// the expected_local_model check on Save and land on that model, so only service-wide edits survive.
export function mergeServerSettings(
  form: AdminSettings,
  baseline: AdminSettings,
  next: AdminSettings,
  keepEdits: KeepEdits | undefined,
  serviceFields: string[],
) {
  if (!keepEdits) {
    return next;
  }
  const serviceOnly = keepEdits === "service" || next.local_model !== baseline.local_model;
  const merged: Record<string, unknown> = { ...next };
  for (const key of Object.keys(form) as Array<keyof AdminSettings>) {
    if (form[key] !== baseline[key] && (!serviceOnly || serviceFields.includes(key))) {
      merged[key] = form[key];
    }
  }
  return merged as AdminSettings;
}
