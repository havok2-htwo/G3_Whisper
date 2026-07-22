import unittest

from backend.genesis_whisper_server_repetition import (
    filter_repeated_patterns,
    repetition_filter_enabled,
)


class RepetitionFilterTests(unittest.TestCase):
    def test_user_example_collapses_phrase_and_name_loops(self) -> None:
        phrase = "dass ich das Gef\u00fchl habe,"
        intro = "Der ist nicht so gut. Ich habe mich nicht mehr so gut verstanden, "
        middle = (
            " Ja, wenn es ein Zoom-Meeting ist, dann lass uns eine Sauerrechnung geben. "
            "Ich dachte, es ist Ihr Telefonat. "
        )
        source = intro + " ".join([phrase] * 28) + middle + " ".join(["Martin?"] * 100)

        expected = intro + phrase + middle + "Martin?"
        self.assertEqual(filter_repeated_patterns(source), expected)

    def test_single_token_threshold_is_five(self) -> None:
        self.assertEqual(filter_repeated_patterns("Martin? Martin? Martin? Martin?"),
                         "Martin? Martin? Martin? Martin?")
        self.assertEqual(filter_repeated_patterns("Martin? Martin? Martin? Martin? Martin?"),
                         "Martin?")

    def test_multi_token_threshold_is_three(self) -> None:
        self.assertEqual(filter_repeated_patterns("nicht gut, nicht gut."),
                         "nicht gut, nicht gut.")
        self.assertEqual(filter_repeated_patterns("nicht gut, nicht gut! nicht gut?"),
                         "nicht gut,")

    def test_normalizes_nfkc_case_whitespace_and_terminal_punctuation(self) -> None:
        source = "\uff2d\uff41\uff52\uff54\uff49\uff4e?\tMARTIN!\nMartin.  martin, Martin;"
        self.assertEqual(filter_repeated_patterns(source), "\uff2d\uff41\uff52\uff54\uff49\uff4e?")

    def test_keeps_first_occurrence_verbatim(self) -> None:
        source = "DAS Gef\u00fchl?! das gef\u00fchl. Das Gef\u00fchl! danach"
        self.assertEqual(filter_repeated_patterns(source), "DAS Gef\u00fchl?! danach")

    def test_does_not_use_fuzzy_matching(self) -> None:
        source = "Martin? Marten? Marton? Marvin? Martyn?"
        self.assertEqual(filter_repeated_patterns(source), source)

    def test_pattern_length_is_limited_to_32_tokens(self) -> None:
        pattern = " ".join(f"wort{index}" for index in range(33))
        source = " | ".join([pattern] * 3)
        self.assertEqual(filter_repeated_patterns(source), source)

    def test_filter_is_idempotent_including_nested_repetitions(self) -> None:
        repeated_word_inside_phrase = "ja ja ja ja ja Ende"
        source = " / ".join([repeated_word_inside_phrase] * 3)
        once = filter_repeated_patterns(source)

        self.assertEqual(once, "ja Ende")
        self.assertEqual(filter_repeated_patterns(once), once)

    def test_empty_and_non_repeating_text_are_unchanged(self) -> None:
        self.assertEqual(filter_repeated_patterns(""), "")
        text = "Ein ganz normales Transkript mit Satzzeichen."
        self.assertEqual(filter_repeated_patterns(text), text)


class RepetitionHeaderTests(unittest.TestCase):
    def test_only_explicit_off_disables_filter(self) -> None:
        self.assertFalse(repetition_filter_enabled("off"))
        self.assertFalse(repetition_filter_enabled(" OFF "))

    def test_missing_empty_and_unknown_values_enable_filter(self) -> None:
        self.assertTrue(repetition_filter_enabled(None))
        self.assertTrue(repetition_filter_enabled(""))
        self.assertTrue(repetition_filter_enabled("on"))
        self.assertTrue(repetition_filter_enabled("false"))


if __name__ == "__main__":
    unittest.main()
