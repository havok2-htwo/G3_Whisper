"""The admin form's merge of fresh server values: what stays typed and what follows the server.

Runs frontend/src/settingsForm.ts in node, which strips the types itself. Skipped without a node that can do that.
"""
import json
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

SETTINGS_FORM = Path(__file__).resolve().parent.parent / "frontend" / "src" / "settingsForm.ts"
SERVICE_FIELDS = ["dia_server_base_url", "huggingface_token"]
BASELINE = {"local_model": "cohere", "batch_max_segments": 16, "batch_wait_time_ms": 500, "dia_server_base_url": ""}


def node_strips_types() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=False).stdout
    match = re.match(r"v(\d+)\.(\d+)", version)
    return bool(match) and (int(match[1]), int(match[2])) >= (22, 18)


def merge(form: dict, next_values: dict, keep_edits: str | None) -> dict:
    script = (
        "const { mergeServerSettings } = await import(process.env.SETTINGS_FORM_URL);"
        "const [form, baseline, next, keepEdits, serviceFields] = JSON.parse(process.env.MERGE_ARGS);"
        "console.log(JSON.stringify(mergeServerSettings(form, baseline, next, keepEdits ?? undefined, serviceFields)));"
    )
    result = subprocess.run(
        ["node", "--no-warnings", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "SETTINGS_FORM_URL": SETTINGS_FORM.as_uri(),
            "MERGE_ARGS": json.dumps([form, BASELINE, next_values, keep_edits, SERVICE_FIELDS]),
        },
    )
    return json.loads(result.stdout)


@unittest.skipUnless(node_strips_types(), "needs node 22.18 or newer")
class MergeServerSettingsTests(unittest.TestCase):
    typed = {**BASELINE, "batch_max_segments": 3, "dia_server_base_url": "http://typed:1"}

    def test_refresh_keeps_typed_values_and_follows_the_server_elsewhere(self) -> None:
        merged = merge(self.typed, {**BASELINE, "batch_wait_time_ms": 777}, "all")
        self.assertEqual(
            merged,
            {**BASELINE, "batch_wait_time_ms": 777, "batch_max_segments": 3, "dia_server_base_url": "http://typed:1"},
        )

    def test_a_model_switched_elsewhere_keeps_only_service_wide_edits(self) -> None:
        parakeet = {**BASELINE, "local_model": "parakeet", "batch_max_segments": 4}
        self.assertEqual(merge(self.typed, parakeet, "all"), {**parakeet, "dia_server_base_url": "http://typed:1"})

    def test_service_mode_drops_model_edits_of_the_same_model(self) -> None:
        self.assertEqual(merge(self.typed, BASELINE, "service"), {**BASELINE, "dia_server_base_url": "http://typed:1"})

    def test_without_keep_edits_the_server_replaces_the_form(self) -> None:
        self.assertEqual(merge(self.typed, BASELINE, None), BASELINE)


if __name__ == "__main__":
    unittest.main()
