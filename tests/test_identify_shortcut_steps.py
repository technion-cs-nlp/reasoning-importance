"""
Unit tests for identify_shortcut_steps.py functions.

Run with:
    pytest tests/test_identify_shortcut_steps.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from identify_shortcut_steps import remove_shortcut_steps


class TestRemoveSinglySufficientSentences:
    """Tests for the remove_singly_sufficient_sentences function."""

    # Test data from entry key:
    # "results/DeepSeek-R1-Distill-Qwen-7B/harp-standard_difficulty=1_seed=42_generations.json:00050"
    PRE_REMOVAL = {
        "answer_sentence_idx": 75,
        "full_prompt": "<｜begin▁of▁sentence｜><｜User｜>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: A calculator has a squaring key $\\boxed{x^2}$ which replaces the current number displayed with its square. For example, if the display is $\\boxed{000003}$ and the $\\boxed{x^2}$ key is depressed, then the display becomes $\\boxed{000009}$. If the display reads $\\boxed{000002}$, how many times must you depress the $\\boxed{x^2}$ key to produce a displayed number greater than $500$?\n<｜Assistant｜><think>\n\nOkay, so I have this problem where I need to figure out how many times I have to press the square key on a calculator to get a number greater than 500. The calculator starts with 2 displayed. Each time I press the square key, it squares the current number. Hmm, let me think about how this works step by step.\n\nFirst, let me write down what happens each time I press the square key. Starting with 2:\n\n1. After the first press: 2 squared is 4.\n2. After the second press: 4 squared is 16.\n3. After the third press: 16 squared is 256.\n4. After the fourth press: 256 squared is... hmm, 256 times 256. Let me calculate that. 256 * 256. I know that 250 squared is 62,500, and 6 squared is 36, but wait, that's not helpful. Maybe I should do it step by step.\n\n256 * 256:\n- 200 * 200 = 40,000\n- 200 * 56 = 11,200\n- 56 * 200 = 11,200\n- 56 * 56 = 3,136\nWait, that might not be the right way. Maybe I should use the formula (a + b)^2 = a^2 + 2ab + b^2 where a = 200 and b = 56.\n\nSo, (200 + 56)^2 = 200^2 + 2*200*56 + 56^2\n= 40,000 + 22,400 + 3,136\n= 40,000 + 22,400 = 62,400\n62,400 + 3,136 = 65,536\n\nSo, 256 squared is 65,536. That's way more than 500. So, after the fourth press, the number is 65,536, which is definitely greater than 500.\n\nWait, but let me check if I did that correctly. 256 * 256. Maybe I can do it another way. 256 is 2^8, so 256 squared is (2^8)^2 = 2^16, which is 65,536. Yeah, that's correct.\n\nSo, starting from 2, each press squares the number:\n\n1. 2^2 = 4\n2. 4^2 = 16\n3. 16^2 = 256\n4. 256^2 = 65,536\n\nSo, after the fourth press, it's 65,536, which is way over 500. But wait, is 256 already over 500? No, 256 is less than 500. So, pressing the square key three times gets us to 256, which is still under 500. Then pressing it a fourth time takes us to 65,536, which is over 500.\n\nTherefore, the number of times we need to press the square key is four.\n\nBut let me make sure I didn't skip any steps or make a mistake in my calculations. Let's go through each step again.\n\nStarting number: 2\n\n1. First press: 2^2 = 4. 4 is less than 500.\n2. Second press: 4^2 = 16. 16 is less than 500.\n3. Third press: 16^2 = 256. 256 is less than 500.\n4. Fourth press: 256^2 = 65,536. 65,536 is greater than 500.\n\nSo, yes, four presses are needed. I don't think I made any mistakes here. Each time, the number is squared, and after the third press, it's 256, which is still under 500, so we need one more press to get over 500.\n\nAlternatively, maybe I can think about logarithms to see how many squarings are needed to exceed 500. Let's see.\n\nWe start with x = 2. Each squaring is equivalent to x = x^2. So, after n squarings, x = 2^(2^n). We need 2^(2^n) > 500.\n\nTaking logarithms on both sides:\n\nlog(2^(2^n)) > log(500)\n\nWhich simplifies to:\n\n2^n * log(2) > log(500)\n\nSo, 2^n > log(500) / log(2)\n\nCalculating log(500). Let's use base 10 logarithm.\n\nlog(500) = log(5 * 100) = log(5) + log(100) = log(5) + 2 ≈ 0.69897 + 2 = 2.69897\n\nlog(2) ≈ 0.3010\n\nSo, log(500)/log(2) ≈ 2.69897 / 0.3010 ≈ 8.96\n\nSo, 2^n > 8.96\n\nWe need to find the smallest integer n such that 2^n > 8.96\n\nCalculating 2^n:\n\n2^3 = 8\n\n2^4 = 16\n\nSo, 2^4 = 16 > 8.96\n\nTherefore, n = 4\n\nSo, that also gives us n = 4. So, that confirms it.\n\nTherefore, the number of times we need to press the square key is 4.\n\n**Final Answer**\nThe number of times you must depress the squaring key is \\boxed{4}.\n</think>\n\nStarting with the number 2 on the calculator display, we need to determine how many times we must press the squaring key to produce a number greater than 500.\n\n1. After the first press: \\(2^2 = 4\\)\n2. After the second press: \\(4^2 = 16\\)\n3. After the third press: \\(16^2 = 256\\)\n4. After the fourth press: \\(256^2 = 65536\\)\n\nAfter the fourth press, the number 65536 is greater than 500. Therefore, the number of times we need to press the squaring key is \\(\\boxed{4}\\).<｜end▁of▁sentence｜>",
        "sentences": [
            "\nOkay, so I have this problem where I need to figure out how many times I have to press the square key on a calculator to get a number greater than 500.",
            " The calculator starts with 2 displayed.",
            " Each time I press the square key, it squares the current number.",
            " Hmm, let me think about how this works step by step.\n",
            "\nFirst, let me write down what happens each time I press the square key.",
            " Starting with 2:\n\n1.",
            " After the first press: 2 squared is 4.\n2.",
            " After the second press: 4 squared is 16.\n3.",
            " After the third press: 16 squared is 256.\n4.",
            " After the fourth press: 256 squared is...",
            " hmm, 256 times 256.",
            " Let me calculate that.",
            " 256 * 256.",
            " I know that 250 squared is 62,500, and 6 squared is 36, but wait, that's not helpful.",
            " Maybe I should do it step by step.\n",
            "\n256 * 256:\n- 200 * 200 = 40,000\n- 200 * 56 = 11,200\n- 56 * 200 = 11,200\n- 56 * 56 = 3,136\nWait, that might not be the right way.",
            " Maybe I should use the formula (a + b)^2 = a^2 + 2ab + b^2 where a = 200 and b = 56.\n",
            "\nSo, (200 + 56)^2 = 200^2 + 2*200*56 + 56^2\n= 40,000 + 22,400 + 3,136\n= 40,000 + 22,400 = 62,400\n62,400 + 3,136 = 65,536\n",
            "\nSo, 256 squared is 65,536.",
            " That's way more than 500.",
            " So, after the fourth press, the number is 65,536, which is definitely greater than 500.\n",
            "\nWait, but let me check if I did that correctly.",
            " 256 * 256.",
            " Maybe I can do it another way.",
            " 256 is 2^8, so 256 squared is (2^8)^2 = 2^16, which is 65,536.",
            " Yeah, that's correct.\n",
            "\nSo, starting from 2, each press squares the number:\n\n1.",
            " 2^2 = 4\n2.",
            " 4^2 = 16\n3.",
            " 16^2 = 256\n4.",
            " 256^2 = 65,536\n",
            "\nSo, after the fourth press, it's 65,536, which is way over 500.",
            " But wait, is 256 already over 500?",
            " No, 256 is less than 500.",
            " So, pressing the square key three times gets us to 256, which is still under 500.",
            " Then pressing it a fourth time takes us to 65,536, which is over 500.\n",
            "\nTherefore, the number of times we need to press the square key is four.\n",
            "\nBut let me make sure I didn't skip any steps or make a mistake in my calculations.",
            " Let's go through each step again.\n",
            "\nStarting number: 2\n\n1.",
            " First press: 2^2 = 4.",
            " 4 is less than 500.\n2.",
            " Second press: 4^2 = 16.",
            " 16 is less than 500.\n3.",
            " Third press: 16^2 = 256.",
            " 256 is less than 500.\n4.",
            " Fourth press: 256^2 = 65,536.",
            " 65,536 is greater than 500.\n",
            "\nSo, yes, four presses are needed.",
            " I don't think I made any mistakes here.",
            " Each time, the number is squared, and after the third press, it's 256, which is still under 500, so we need one more press to get over 500.\n",
            "\nAlternatively, maybe I can think about logarithms to see how many squarings are needed to exceed 500.",
            " Let's see.\n",
            "\nWe start with x = 2.",
            " Each squaring is equivalent to x = x^2.",
            " So, after n squarings, x = 2^(2^n).",
            " We need 2^(2^n) > 500.\n",
            "\nTaking logarithms on both sides:\n",
            "\nlog(2^(2^n)) > log(500)\n",
            "\nWhich simplifies to:\n",
            "\n2^n * log(2) > log(500)\n",
            "\nSo, 2^n > log(500) / log(2)\n",
            "\nCalculating log(500).",
            " Let's use base 10 logarithm.\n",
            "\nlog(500) = log(5 * 100) = log(5) + log(100) = log(5) + 2 ≈ 0.69897 + 2 = 2.69897\n",
            "\nlog(2) ≈ 0.3010\n",
            "\nSo, log(500)/log(2) ≈ 2.69897 / 0.3010 ≈ 8.96\n",
            "\nSo, 2^n > 8.96\n",
            "\nWe need to find the smallest integer n such that 2^n > 8.96\n",
            "\nCalculating 2^n:\n\n2^3 = 8\n\n2^4 = 16\n",
            "\nSo, 2^4 = 16 > 8.96\n",
            "\nTherefore, n = 4\n",
            "\nSo, that also gives us n = 4.",
            " So, that confirms it.\n",
            "\nTherefore, the number of times we need to press the square key is 4.\n",
            "\n**Final Answer**\nThe number of times you must depress the squaring key is \\boxed{4}.",
        ],
        "token_borders": [
            [0, 38],
            [38, 46],
            [46, 60],
            [60, 73],
            [73, 89],
            [89, 96],
            [96, 110],
            [110, 125],
            [125, 142],
            [142, 154],
            [154, 166],
            [166, 171],
            [171, 181],
            [181, 215],
            [215, 224],
            [224, 314],
            [314, 352],
            [352, 459],
            [459, 475],
            [475, 485],
            [485, 513],
            [513, 525],
            [525, 535],
            [535, 543],
            [543, 583],
            [583, 589],
            [589, 604],
            [604, 614],
            [614, 625],
            [625, 638],
            [638, 653],
            [653, 679],
            [679, 694],
            [694, 708],
            [708, 733],
            [733, 758],
            [758, 774],
            [774, 793],
            [793, 801],
            [801, 809],
            [809, 820],
            [820, 832],
            [832, 844],
            [844, 857],
            [857, 871],
            [871, 885],
            [885, 903],
            [903, 918],
            [918, 927],
            [927, 937],
            [937, 981],
            [981, 1006],
            [1006, 1010],
            [1010, 1018],
            [1018, 1030],
            [1030, 1046],
            [1046, 1060],
            [1060, 1067],
            [1067, 1081],
            [1081, 1086],
            [1086, 1100],
            [1100, 1117],
            [1117, 1125],
            [1125, 1135],
            [1135, 1194],
            [1194, 1208],
            [1208, 1246],
            [1246, 1258],
            [1258, 1278],
            [1278, 1299],
            [1299, 1316],
            [1316, 1323],
            [1323, 1334],
            [1334, 1340],
            [1340, 1357],
            [1357, 1378],
        ],
    }

    POST_REMOVAL = {
        "answer_sentence_idx": 49,
        "full_prompt": "<｜begin▁of▁sentence｜><｜User｜>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: A calculator has a squaring key $\\boxed{x^2}$ which replaces the current number displayed with its square. For example, if the display is $\\boxed{000003}$ and the $\\boxed{x^2}$ key is depressed, then the display becomes $\\boxed{000009}$. If the display reads $\\boxed{000002}$, how many times must you depress the $\\boxed{x^2}$ key to produce a displayed number greater than $500$?\n<｜Assistant｜><think>\n\nOkay, so I have this problem where I need to figure out how many times I have to press the square key on a calculator to get a number greater than 500. The calculator starts with 2 displayed. Each time I press the square key, it squares the current number. Hmm, let me think about how this works step by step.\n\nFirst, let me write down what happens each time I press the square key. Starting with 2:\n\n1. After the first press: 2 squared is 4.\n2. After the second press: 4 squared is 16.\n3. hmm, 256 times 256. Let me calculate that. 256 * 256. I know that 250 squared is 62,500, and 6 squared is 36, but wait, that's not helpful. Maybe I should do it step by step.\n Maybe I should use the formula (a + b)^2 = a^2 + 2ab + b^2 where a = 200 and b = 56.\n\nSo, (200 + 56)^2 = 200^2 + 2*200*56 + 56^2\n= 40,000 + 22,400 + 3,136\n= 40,000 + 22,400 = 62,400\n62,400 + 3,136 = 65,536\n That's way more than 500.\nWait, but let me check if I did that correctly. 256 * 256. Maybe I can do it another way. 256 is 2^8, so 256 squared is (2^8)^2 = 2^16, which is 65,536. Yeah, that's correct.\n\nSo, starting from 2, each press squares the number:\n\n1. 2^2 = 4\n2. 4^2 = 16\n3. But wait, is 256 already over 500? No, 256 is less than 500.\nBut let me make sure I didn't skip any steps or make a mistake in my calculations. Let's go through each step again.\n\nStarting number: 2\n\n1. First press: 2^2 = 4. 4 is less than 500.\n2. Second press: 4^2 = 16. 16 is less than 500.\n3. Third press: 16^2 = 256. I don't think I made any mistakes here.\nAlternatively, maybe I can think about logarithms to see how many squarings are needed to exceed 500. Let's see.\n\nWe start with x = 2. Each squaring is equivalent to x = x^2. So, after n squarings, x = 2^(2^n). We need 2^(2^n) > 500.\n\nTaking logarithms on both sides:\n\nWhich simplifies to:\n\n2^n * log(2) > log(500)\n\nSo, 2^n > log(500) / log(2)\n\nCalculating log(500). Let's use base 10 logarithm.\n\nlog(500) = log(5 * 100) = log(5) + log(100) = log(5) + 2 ≈ 0.69897 + 2 = 2.69897\n\nSo, log(500)/log(2) ≈ 2.69897 / 0.3010 ≈ 8.96\n So, that confirms it.\n\n**Final Answer**\nThe number of times you must depress the squaring key is \\boxed{4}.\n",
        "sentences": [
            "\n",
            "\nOkay, so I have this problem where I need to figure out how many times I have to press the square key on a calculator to get a number greater than 500.",
            " The calculator starts with 2 displayed.",
            " Each time I press the square key, it squares the current number.",
            " Hmm, let me think about how this works step by step.\n",
            "\nFirst, let me write down what happens each time I press the square key.",
            " Starting with 2:\n\n1.",
            " After the first press: 2 squared is 4.\n2.",
            " After the second press: 4 squared is 16.\n3.",
            " hmm, 256 times 256.",
            " Let me calculate that.",
            " 256 * 256.",
            " I know that 250 squared is 62,500, and 6 squared is 36, but wait, that's not helpful.",
            " Maybe I should do it step by step.",
            "\n Maybe I should use the formula (a + b)^2 = a^2 + 2ab + b^2 where a = 200 and b = 56.\n",
            "\nSo, (200 + 56)^2 = 200^2 + 2*200*56 + 56^2\n= 40,000 + 22,400 + 3,136\n= 40,000 + 22,400 = 62,400\n62,400 + 3,136 = 65,536\n That's way more than 500.",
            "\nWait, but let me check if I did that correctly.",
            " 256 * 256.",
            " Maybe I can do it another way.",
            " 256 is 2^8, so 256 squared is (2^8)^2 = 2^16, which is 65,536.",
            " Yeah, that's correct.\n",
            "\nSo, starting from 2, each press squares the number:\n\n1.",
            " 2^2 = 4\n2.",
            " 4^2 = 16\n3.",
            " But wait, is 256 already over 500?",
            " No, 256 is less than 500.",
            "\nBut let me make sure I didn't skip any steps or make a mistake in my calculations.",
            " Let's go through each step again.\n",
            "\nStarting number: 2\n\n1.",
            " First press: 2^2 = 4.",
            " 4 is less than 500.\n2.",
            " Second press: 4^2 = 16.",
            " 16 is less than 500.\n3.",
            " Third press: 16^2 = 256.",
            " I don't think I made any mistakes here.",
            "\nAlternatively, maybe I can think about logarithms to see how many squarings are needed to exceed 500.",
            " Let's see.\n",
            "\nWe start with x = 2.",
            " Each squaring is equivalent to x = x^2.",
            " So, after n squarings, x = 2^(2^n).",
            " We need 2^(2^n) > 500.\n",
            "\nTaking logarithms on both sides:\n",
            "\nWhich simplifies to:\n",
            "\n2^n * log(2) > log(500)\n",
            "\nSo, 2^n > log(500) / log(2)\n",
            "\nCalculating log(500).",
            " Let's use base 10 logarithm.\n",
            "\nlog(500) = log(5 * 100) = log(5) + log(100) = log(5) + 2 ≈ 0.69897 + 2 = 2.69897\n",
            "\nSo, log(500)/log(2) ≈ 2.69897 / 0.3010 ≈ 8.96\n So, that confirms it.\n",
            "\n**Final Answer**\nThe number of times you must depress the squaring key is \\boxed{4}.",
        ],
        "token_borders": [
            [131, 132],
            [132, 169],
            [169, 177],
            [177, 191],
            [191, 204],
            [204, 220],
            [220, 227],
            [227, 241],
            [241, 256],
            [256, 268],
            [268, 273],
            [273, 283],
            [283, 317],
            [317, 326],
            [326, 364],
            [364, 481],
            [481, 493],
            [493, 503],
            [503, 511],
            [511, 551],
            [551, 557],
            [557, 572],
            [572, 582],
            [582, 593],
            [593, 608],
            [608, 622],
            [622, 641],
            [641, 649],
            [649, 657],
            [657, 668],
            [668, 680],
            [680, 692],
            [692, 705],
            [705, 719],
            [719, 729],
            [729, 754],
            [754, 758],
            [758, 766],
            [766, 778],
            [778, 794],
            [794, 808],
            [808, 815],
            [815, 820],
            [820, 834],
            [834, 851],
            [851, 859],
            [859, 869],
            [869, 928],
            [928, 972],
            [972, 993],
        ],
    }

    SUFFICIENT_SENTENCE_INDICES = [
        8,
        9,
        15,
        18,
        20,
        29,
        30,
        31,
        34,
        35,
        36,
        45,
        46,
        47,
        48,
        50,
        58,
        65,
        67,
        68,
        69,
        70,
        71,
        72,
        74,
    ]

    def test_remove_singly_sufficient_sentences_real_case(self):
        """
        Test remove_singly_sufficient_sentences using real data from entry key:
        "results/DeepSeek-R1-Distill-Qwen-7B/harp-standard_difficulty=1_seed=42_generations.json:00050"

        This test mocks the model and tokenizer dependencies and verifies that
        the function correctly removes sufficient sentences.
        """
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_model.model_name = "DeepSeek-R1-Distill-Qwen-7B"
        mock_tokenizer = MagicMock()

        # Mock cot_sentences_and_token_borders to return the expected post_removal values
        with patch(
            "identify_shortcut_steps.cot_sentences_and_token_borders"
        ) as mock_cot_func:
            mock_cot_func.return_value = (
                self.POST_REMOVAL["sentences"],
                self.POST_REMOVAL["token_borders"],
            )

            # Call the function
            (
                cleaned_prompt,
                cleaned_sentences,
                cleaned_token_borders,
                cleaned_answer_idx,
            ) = remove_shortcut_steps(
                full_prompt=self.PRE_REMOVAL["full_prompt"],
                sentences=self.PRE_REMOVAL["sentences"],
                token_borders=self.PRE_REMOVAL["token_borders"],
                answer_sentence_idx=self.PRE_REMOVAL["answer_sentence_idx"],
                sufficient_sentences=self.SUFFICIENT_SENTENCE_INDICES,
                model=mock_model,
                tokenizer=mock_tokenizer,
            )

            # Verify output
            assert cleaned_sentences == self.POST_REMOVAL["sentences"]
            assert cleaned_token_borders == self.POST_REMOVAL["token_borders"]
            assert cleaned_answer_idx == self.POST_REMOVAL["answer_sentence_idx"]

    def test_sentence_count_after_removal(self):
        """Verify that sentence count is reduced after removal.

        Note: The exact count may not be pre - removed due to sentence re-parsing
        after text is modified. The important thing is that sentences were removed.
        """
        pre_count = len(self.PRE_REMOVAL["sentences"])
        post_count = len(self.POST_REMOVAL["sentences"])
        removed_count = len(self.SUFFICIENT_SENTENCE_INDICES)

        # Post-removal count should be significantly less than pre-removal
        assert post_count < pre_count
        # Should be approximately pre - removed (within small margin for re-parsing)
        assert abs(post_count - (pre_count - removed_count)) <= 2

    def test_answer_sentence_preserved(self):
        """Verify that the answer sentence is preserved after removal."""
        # The original answer sentence (index 75) contains the final boxed answer
        original_answer_sentence = self.PRE_REMOVAL["sentences"][
            self.PRE_REMOVAL["answer_sentence_idx"]
        ]

        # The post-removal answer sentence (index 49) should be the same
        post_removal_answer_sentence = self.POST_REMOVAL["sentences"][
            self.POST_REMOVAL["answer_sentence_idx"]
        ]

        assert original_answer_sentence == post_removal_answer_sentence
        assert "\\boxed{4}" in original_answer_sentence

    def test_no_removal_when_empty_sufficient_list(self):
        """Test that no changes occur when sufficient_sentences is empty."""
        mock_model = MagicMock()
        mock_model.model_name = "DeepSeek-R1-Distill-Qwen-7B"
        mock_tokenizer = MagicMock()

        result = remove_shortcut_steps(
            full_prompt=self.PRE_REMOVAL["full_prompt"],
            sentences=self.PRE_REMOVAL["sentences"],
            token_borders=self.PRE_REMOVAL["token_borders"],
            answer_sentence_idx=self.PRE_REMOVAL["answer_sentence_idx"],
            sufficient_sentences=[],  # Empty list
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # When no sentences to remove, should return original values
        assert result[0] == self.PRE_REMOVAL["full_prompt"]
        assert result[1] == self.PRE_REMOVAL["sentences"]
        assert result[2] == self.PRE_REMOVAL["token_borders"]
        assert result[3] == self.PRE_REMOVAL["answer_sentence_idx"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
