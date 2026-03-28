import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app


class AppHelpersTests(unittest.TestCase):
    def test_load_settings_reads_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "openai-secret-key": "test-key",
                        "openai-model": "test-model",
                    }
                ),
                encoding="utf-8",
            )

            settings = app.load_settings(config_path)

        self.assertEqual(settings["api_key"], "test-key")
        self.assertEqual(settings["model"], "test-model")

    def test_load_settings_prefers_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "openai-secret-key": "config-key",
                        "openai-model": "config-model",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "env-key", "OPENAI_MODEL": "env-model"},
                clear=False,
            ):
                settings = app.load_settings(config_path)

        self.assertEqual(settings["api_key"], "env-key")
        self.assertEqual(settings["model"], "env-model")

    def test_build_context_skips_blank_reviews(self) -> None:
        context = app.build_context([" first review ", "", "   ", "second review"])

        self.assertIn("- first review", context)
        self.assertIn("- second review", context)
        self.assertNotIn("- \n", context)

    def test_build_offline_answer_includes_reason_and_reviews(self) -> None:
        answer = app.build_offline_answer(
            "Any question?",
            [" First matching review. ", "Second matching review."],
            reason="No se encontró una API key de OpenAI.",
        )

        self.assertIn("Modo local activado", answer)
        self.assertIn("Motivo: No se encontró una API key de OpenAI.", answer)
        self.assertIn("1. First matching review.", answer)

    def test_generate_chat_response_falls_back_to_local_context_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(
                    app,
                    "retrieve_reviews",
                    return_value=["Relevant local review.", "Another review."],
                ):
                    answer = app.generate_chat_response(
                        "Tell me about this car",
                        config_path=config_path,
                    )

        self.assertIn("Modo local activado", answer)
        self.assertIn("No se encontró una API key de OpenAI.", answer)
        self.assertIn("Relevant local review.", answer)


if __name__ == "__main__":
    unittest.main()
