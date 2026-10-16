"""Settings presets: built-in GPU tiers, card matching for the two-GPU tier, saved presets and the admin routes.

Runs without a GPU: the detected cards are faked, and the presets file lives in a temporary directory so a test
never writes into the real logs/ directory.
"""
import os
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_whisper_server_admin as admin
from backend import genesis_whisper_server_presets as presets
from backend import genesis_whisper_server_storage as storage

TIER_NAMES = ["32GB VRAM", "24GB VRAM", "24+8 VRAM"]


class _PresetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="genesis-whisper-presets-tests-")
        self.presets_file = os.path.join(self.temp_dir.name, "genesis_whisper_presets.json")
        self.patches = [
            mock.patch.object(presets, "PRESETS_FILE", self.presets_file),
            # Never touch the real settings file either, even if a route under test would save settings.
            mock.patch.object(storage, "LOGS_DIR", self.temp_dir.name),
            mock.patch.object(storage, "SETTINGS_FILE", os.path.join(self.temp_dir.name, "settings.json")),
        ]
        for patcher in self.patches:
            patcher.start()
        self.cards()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.temp_dir.cleanup()

    def cards(self, *cards_mib: int) -> None:
        patcher = mock.patch.object(presets, "_cuda_cards_mib", return_value=list(cards_mib))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tier(self, name: str) -> dict:
        return next(preset for preset in presets.resolve_presets() if preset["name"] == name)


class BuiltinPresetTests(_PresetTestCase):
    def test_builtin_tiers_come_first_in_tier_order(self) -> None:
        self.cards(32607)
        names = [preset["name"] for preset in presets.resolve_presets()]
        self.assertEqual(names[: len(TIER_NAMES)], TIER_NAMES)

    def test_whisper_has_no_8_plus_24_tier(self) -> None:
        self.assertNotIn("8+24 VRAM", presets.BUILTIN_PRESETS)

    def test_builtin_values_are_valid_settings(self) -> None:
        for name in TIER_NAMES:
            with self.subTest(name=name):
                values = presets.BUILTIN_PRESETS[name]["values"]
                self.assertLessEqual(set(values), set(presets.PRESET_KEYS))
                normalized = storage.normalize_settings(values)
                self.assertEqual(
                    {key: normalized[key] for key in values}, values, "a built-in value would be clamped or rejected"
                )

    def test_preset_keys_hold_no_secrets_or_paths(self) -> None:
        for key in ("huggingface_token", "dia_api_key", "dia_server_base_url", "local_model_cache_path"):
            self.assertNotIn(key, presets.PRESET_KEYS)
        self.assertLessEqual(set(presets.PRESET_KEYS), set(storage.DEFAULT_SETTINGS))

    def test_two_gpu_tier_picks_the_24gb_card_wherever_it_sits(self) -> None:
        for cards, expected in (((24576, 8192), "cuda:0"), ((8192, 24576), "cuda:1"), ((7982, 24268), "cuda:1")):
            with self.subTest(cards=cards):
                self.cards(*cards)
                tier = self.tier("24+8 VRAM")
                self.assertEqual(tier["values"]["local_gpu_device"], expected)
                self.assertIsNone(tier["note"])

    def test_two_gpu_tier_without_a_matching_card_leaves_the_gpu_alone(self) -> None:
        self.cards(32607)
        tier = self.tier("24+8 VRAM")
        self.assertNotIn("local_gpu_device", tier["values"])
        self.assertIn("No 24 GB card", tier["note"])

    def test_two_gpu_tier_without_any_card_explains_it(self) -> None:
        tier = self.tier("24+8 VRAM")
        self.assertNotIn("local_gpu_device", tier["values"])
        self.assertIn("cards: none", tier["note"])

    def test_single_card_tiers_never_set_the_gpu(self) -> None:
        self.cards(24576, 8192)
        for preset in presets.resolve_presets():
            if preset["name"] in ("32GB VRAM", "24GB VRAM"):
                self.assertNotIn("local_gpu_device", preset["values"])
                self.assertIsNone(preset["note"])

    def test_resolving_does_not_mutate_the_builtin_values(self) -> None:
        self.cards(24576)
        presets.resolve_presets()
        self.assertNotIn("local_gpu_device", presets.BUILTIN_PRESETS["24+8 VRAM"]["values"])


class UserPresetTests(_PresetTestCase):
    def test_saved_preset_keeps_only_preset_keys_normalised(self) -> None:
        saved = presets.save_user_preset(
            "  my   tier ",
            {
                "batch_max_segments": 0,
                "batch_wait_time_ms": "250",
                "huggingface_token": "hf_secret",
                "dia_api_key": "dia_secret",
                "local_model_cache_path": "D:\\models",
                "nonsense": 1,
                "local_model_precision": "BF16",
            },
        )
        self.assertEqual(saved["name"], "my tier")
        self.assertEqual(
            saved["values"], {"batch_max_segments": 1, "batch_wait_time_ms": 250, "local_model_precision": "bf16"}
        )
        listed = [preset for preset in presets.resolve_presets() if not preset["built_in"]]
        self.assertEqual([(preset["name"], preset["values"]) for preset in listed], [("my tier", saved["values"])])
        with open(self.presets_file, "r", encoding="utf-8") as file_obj:
            stored = file_obj.read()
        self.assertNotIn("hf_secret", stored)
        self.assertNotIn("dia_secret", stored)
        self.assertNotIn("models", stored)

    def test_resaving_keeps_created_at_and_sorts_by_name(self) -> None:
        first = presets.save_user_preset("beta", {"batch_max_segments": 8})
        presets.save_user_preset("Alpha", {"batch_max_segments": 4})
        again = presets.save_user_preset("beta", {"batch_max_segments": 12})
        self.assertEqual(again["created_at"], first["created_at"])
        user = [preset["name"] for preset in presets.resolve_presets() if not preset["built_in"]]
        self.assertEqual(user, ["Alpha", "beta"])
        self.assertEqual(self.tier("beta")["values"], {"batch_max_segments": 12})

    def test_builtin_names_are_protected(self) -> None:
        with self.assertRaises(ValueError):
            presets.save_user_preset("32GB VRAM", {"batch_max_segments": 8})
        with self.assertRaises(ValueError):
            presets.delete_user_preset("24GB VRAM")

    def test_delete_saved_preset(self) -> None:
        presets.save_user_preset("tmp", {"batch_max_segments": 8})
        self.assertTrue(presets.delete_user_preset("tmp"))
        self.assertFalse(presets.delete_user_preset("tmp"))
        self.assertTrue(all(preset["built_in"] for preset in presets.resolve_presets()))

    def test_empty_or_long_names_and_empty_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            presets.save_user_preset("   ", {"batch_max_segments": 8})
        with self.assertRaises(ValueError):
            presets.save_user_preset("x" * 65, {"batch_max_segments": 8})
        with self.assertRaises(ValueError):
            presets.save_user_preset("only unknown keys", {"nonsense": 1})

    def test_corrupt_presets_file_lists_only_builtins(self) -> None:
        with open(self.presets_file, "w", encoding="utf-8") as file_obj:
            file_obj.write("{not json")
        self.assertEqual([preset["name"] for preset in presets.resolve_presets()], TIER_NAMES)

    def test_file_entry_under_a_builtin_name_is_ignored(self) -> None:
        with open(self.presets_file, "w", encoding="utf-8") as file_obj:
            file_obj.write('{"24GB VRAM": {"values": {"batch_max_segments": 2}}}')
        self.assertEqual([preset["name"] for preset in presets.resolve_presets()], TIER_NAMES)
        self.assertEqual(self.tier("24GB VRAM")["values"], presets.BUILTIN_PRESETS["24GB VRAM"]["values"])


class AdminPresetRouteTests(_PresetTestCase):
    def setUp(self) -> None:
        super().setUp()
        list_patch = mock.patch.object(admin, "list_model_statuses", return_value=[])
        list_patch.start()
        self.addCleanup(list_patch.stop)
        self.app = admin.create_admin_api(FastAPI())
        self.app.dependency_overrides[admin.require_admin] = lambda: {"username": "admin"}
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        super().tearDown()

    def test_admin_preset_routes(self) -> None:
        self.cards(32607)
        listed = self.client.get("/api/admin/settings/presets").json()["presets"]
        self.assertEqual([preset["name"] for preset in listed], TIER_NAMES)

        response = self.client.put(
            "/api/admin/settings/presets/lab",
            json={"values": {"batch_max_segments": 8, "local_gpu_device": "cuda:1", "huggingface_token": "hf_x"}},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["preset"]["values"], {"batch_max_segments": 8, "local_gpu_device": "cuda:1"})

        self.assertEqual(
            self.client.put("/api/admin/settings/presets/32GB VRAM", json={"values": {"batch_max_segments": 8}}).status_code,
            400,
        )
        self.assertEqual(self.client.delete("/api/admin/settings/presets/24GB VRAM").status_code, 400)
        self.assertEqual(self.client.delete("/api/admin/settings/presets/lab").status_code, 200)
        self.assertEqual(self.client.delete("/api/admin/settings/presets/lab").status_code, 404)

    def test_preset_routes_require_admin(self) -> None:
        self.app.dependency_overrides.clear()
        self.assertIn(self.client.get("/api/admin/settings/presets").status_code, (401, 403))
        self.assertIn(
            self.client.put("/api/admin/settings/presets/x", json={"values": {"batch_max_segments": 8}}).status_code,
            (401, 403),
        )
        self.assertFalse(os.path.exists(self.presets_file))


if __name__ == "__main__":
    unittest.main()
