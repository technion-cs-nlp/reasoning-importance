"""
Unit tests for cot_utils.py functions.

Run with:
    pytest tests/test_cot_utils.py -v

Or run all tests:
    pytest tests/ -v
"""

import pytest
import torch
from cot_utils import (
    extract_cot_from_output,
    _split_solution_into_sentences,
    search_answer,
    sentence_states_answer,
    apply_step_mask_to_cot,
    are_answers_equivalent,
    normalize_latex,
    extract_boxed_answers,
    try_parse_number,
    normalize_symbolic_addition,
    normalize_tuple,
    AnswerError,
)

# =============================================================================
# Tests for extract_cot_from_output
# =============================================================================


class TestExtractCotFromOutput:
    """Tests for the extract_cot_from_output function."""

    # GPT-OSS test cases: (prompt, expected_cot)
    GPT_OSS_CASES = [
        # Analysis tag
        (
            "<|start|>analysis<|message|>This is the thinking process.<|end|>Final answer.",
            "This is the thinking process.",
        ),
        # Commentary tag
        (
            "<|channel|>commentary<|message|>Thinking here.<|end|>Answer.",
            "Thinking here.",
        ),
        # Bare <|start|>...<|end|> block before <|start|>assistant<|channel|>final
        (
            "<|start|>system<|message|>You are ChatGPT.<|end|><|start|>user<|message|>Solve the problem.<|end|><|start|>We have a problem. So answer 250. Provide in boxed form.<|end|><|start|>assistant<|channel|>final<|message|>\\boxed{250}<|return|>",
            "We have a problem. So answer 250. Provide in boxed form.",
        ),
        # Bare <|start|>... block without closing tag
        (
            "<|start|>system<|message|>You are ChatGPT.<|end|><|start|>user<|message|>Solve the problem.<|end|><|start|>We have a problem. So answer 250. Provide in boxed form.",
            "We have a problem. So answer 250. Provide in boxed form.",
        ),
        # Full prompt
        (
            '<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-10-09\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: [asy] unitsize(13); draw((0,0)--(20,0)); draw((0,0)--(0,15)); draw((0,3)--(-1,3)); draw((0,6)--(-1,6)); draw((0,9)--(-1,9)); draw((0,12)--(-1,12)); draw((0,15)--(-1,15)); fill((2,0)--(2,15)--(3,15)--(3,0)--cycle,black); fill((4,0)--(4,12)--(5,12)--(5,0)--cycle,black); fill((6,0)--(6,9)--(7,9)--(7,0)--cycle,black); fill((8,0)--(8,9)--(9,9)--(9,0)--cycle,black); fill((10,0)--(10,15)--(11,15)--(11,0)--cycle,black); label("A",(2.5,-.5),S); label("B",(4.5,-.5),S); label("C",(6.5,-.5),S); label("D",(8.5,-.5),S); label("F",(10.5,-.5),S); label("Grade",(15,-.5),S); label("$1$",(-1,3),W); label("$2$",(-1,6),W); label("$3$",(-1,9),W); label("$4$",(-1,12),W); label("$5$",(-1,15),W); [/asy] The bar graph shows the grades in a mathematics class for the last grading period. If A, B, C, and D are satisfactory grades, what fraction of the grades shown in the graph are satisfactory?\n<|end|><|start|>analysis<|message|>We need to parse the problem. There\'s a bar graph with bars labeled A, B, C, D, F. The y-axis labeled 1-5. The bars heights: A: 5? Wait the graph shows bars of varying heights. Let\'s interpret: The y-axis values 1-5 correspond to grade categories? Actually typical grading: 1-5 maybe numeric? But the bars show heights: A bar from 0 to 15? Wait the graph uses units: The y-axis increments 3 units? Let\'s analyze.\n\nThe graph: draw((0,0)--(20,0)); horizontal axis length 20 units. draw((0,0)--(0,15)); vertical axis length 15 units. Then vertical tick marks at y=3,6,9,12,15 labeled 1-5. So each tick is 3 units. So the vertical axis is 5 units high, each representing a grade level? Actually the labels 1-5 correspond to grade categories? But the bars are filled rectangles: A: from x=2 to 3, height 15 (full). B: from x=4 to 5, height 12. C: from x=6 to 7, height 9. D: from x=8 to 9, height 9. F: from x=10 to 11, height 15? Wait F bar height 15? Actually the fill for F: (10,0)--(10,15)--(11,15)--(11,0). So F bar height 15. So F has 5 units? Wait vertical axis 15 units corresponds to 5? So F has 5? But F is unsatisfactory grade. But the bar for F is full height 15, meaning 5? That seems odd. Let\'s double-check: The bars for A and F are full height 15. B is 12, C and D are 9. So the heights correspond to number of students? Actually the y-axis labeled 1-5 maybe number of students? But the bars show heights 15,12,9,9,15. That seems like counts of students? But the y-axis labeled 1-5 maybe number of students? Wait the y-axis labels 1-5 at ticks 3,6,9,12,15. So each tick corresponds to 1 student? That would mean the bar heights represent number of students times 3 units per student. So A has 5 students, B has 4, C has 3, D has 3, F has 5. But that seems plausible: The graph shows the number of students in each grade. The y-axis labeled 1-5 indicates number of students. So each bar height corresponds to number of students. So A: 5 students, B: 4, C: 3, D: 3, F: 5. But wait F has 5? That would mean 5 students got F. But the bar for F is full height 15, same as A. So yes.\n\nBut the problem: "The bar graph shows the grades in a mathematics class for the last grading period. If A, B, C, and D are satisfactory grades, what fraction of the grades shown in the graph are satisfactory?" So we need to compute fraction of total grades that are satisfactory (A-D). So we need total number of grades: sum of all bars: A 5 + B 4 + C 3 + D 3 + F 5 = 20. Satisfactory: A+B+C+D = 5+4+3+3 = 15. So fraction = 15/20 = 3/4. So answer: 3/4. But we need to express as fraction in simplest form: 3/4. So final answer: \\boxed{\\frac{3}{4}}. Let\'s double-check if any nuance: The bars might represent number of students per grade. The y-axis labeled 1-5 indicates number of students. So yes.\n\nBut we should confirm that the bar heights correspond to 5,4,3,3,5. Let\'s compute: A: height 15 units -> 5. B: height 12 -> 4. C: height 9 -> 3. D: height 9 -> 3. F: height 15 -> 5. So total 20. Satisfactory: 5+4+3+3=15. So fraction 15/20=3/4. Good.\n\nThus final answer: \\boxed{\\frac{3}{4}}.<|end|><|start|>final<|message|>The y\u2011axis is labeled 1\u20135, so each 3\u2011unit increment represents one student.  \n- A: height 15 \u2192 5 students  \n- B: height 12 \u2192 4 students  \n- C: height 9  \u2192 3 students  \n- D: height 9  \u2192 3 students  \n- F: height 15 \u2192 5 students  \n\nTotal students = \\(5+4+3+3+5 = 20\\).  \nSatisfactory grades (A, B, C, D) = \\(5+4+3+3 = 15\\).\n\n\\[\n\\text{Fraction satisfactory} = \\frac{15}{20} = \\frac{3}{4}\n\\]\n\n\\[\n\\boxed{\\frac{3}{4}}\n\\]<|return|>',
            "We need to parse the problem. There's a bar graph with bars labeled A, B, C, D, F. The y-axis labeled 1-5. The bars heights: A: 5? Wait the graph shows bars of varying heights. Let's interpret: The y-axis values 1-5 correspond to grade categories? Actually typical grading: 1-5 maybe numeric? But the bars show heights: A bar from 0 to 15? Wait the graph uses units: The y-axis increments 3 units? Let's analyze.\n\nThe graph: draw((0,0)--(20,0)); horizontal axis length 20 units. draw((0,0)--(0,15)); vertical axis length 15 units. Then vertical tick marks at y=3,6,9,12,15 labeled 1-5. So each tick is 3 units. So the vertical axis is 5 units high, each representing a grade level? Actually the labels 1-5 correspond to grade categories? But the bars are filled rectangles: A: from x=2 to 3, height 15 (full). B: from x=4 to 5, height 12. C: from x=6 to 7, height 9. D: from x=8 to 9, height 9. F: from x=10 to 11, height 15? Wait F bar height 15? Actually the fill for F: (10,0)--(10,15)--(11,15)--(11,0). So F bar height 15. So F has 5 units? Wait vertical axis 15 units corresponds to 5? So F has 5? But F is unsatisfactory grade. But the bar for F is full height 15, meaning 5? That seems odd. Let's double-check: The bars for A and F are full height 15. B is 12, C and D are 9. So the heights correspond to number of students? Actually the y-axis labeled 1-5 maybe number of students? But the bars show heights 15,12,9,9,15. That seems like counts of students? But the y-axis labeled 1-5 maybe number of students? Wait the y-axis labels 1-5 at ticks 3,6,9,12,15. So each tick corresponds to 1 student? That would mean the bar heights represent number of students times 3 units per student. So A has 5 students, B has 4, C has 3, D has 3, F has 5. But that seems plausible: The graph shows the number of students in each grade. The y-axis labeled 1-5 indicates number of students. So each bar height corresponds to number of students. So A: 5 students, B: 4, C: 3, D: 3, F: 5. But wait F has 5? That would mean 5 students got F. But the bar for F is full height 15, same as A. So yes.\n\nBut the problem: \"The bar graph shows the grades in a mathematics class for the last grading period. If A, B, C, and D are satisfactory grades, what fraction of the grades shown in the graph are satisfactory?\" So we need to compute fraction of total grades that are satisfactory (A-D). So we need total number of grades: sum of all bars: A 5 + B 4 + C 3 + D 3 + F 5 = 20. Satisfactory: A+B+C+D = 5+4+3+3 = 15. So fraction = 15/20 = 3/4. So answer: 3/4. But we need to express as fraction in simplest form: 3/4. So final answer: \\boxed{\\frac{3}{4}}. Let's double-check if any nuance: The bars might represent number of students per grade. The y-axis labeled 1-5 indicates number of students. So yes.\n\nBut we should confirm that the bar heights correspond to 5,4,3,3,5. Let's compute: A: height 15 units -> 5. B: height 12 -> 4. C: height 9 -> 3. D: height 9 -> 3. F: height 15 -> 5. So total 20. Satisfactory: 5+4+3+3=15. So fraction 15/20=3/4. Good.\n\nThus final answer: \\boxed{\\frac{3}{4}}.",
        ),
        # Full prompt, commentary channel
        (
            '<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-10-09\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: [asy] unitsize(13); draw((0,0)--(20,0)); draw((0,0)--(0,15)); draw((0,3)--(-1,3)); draw((0,6)--(-1,6)); draw((0,9)--(-1,9)); draw((0,12)--(-1,12)); draw((0,15)--(-1,15)); fill((2,0)--(2,15)--(3,15)--(3,0)--cycle,black); fill((4,0)--(4,12)--(5,12)--(5,0)--cycle,black); fill((6,0)--(6,9)--(7,9)--(7,0)--cycle,black); fill((8,0)--(8,9)--(9,9)--(9,0)--cycle,black); fill((10,0)--(10,15)--(11,15)--(11,0)--cycle,black); label("A",(2.5,-.5),S); label("B",(4.5,-.5),S); label("C",(6.5,-.5),S); label("D",(8.5,-.5),S); label("F",(10.5,-.5),S); label("Grade",(15,-.5),S); label("$1$",(-1,3),W); label("$2$",(-1,6),W); label("$3$",(-1,9),W); label("$4$",(-1,12),W); label("$5$",(-1,15),W); [/asy] The bar graph shows the grades in a mathematics class for the last grading period. If A, B, C, and D are satisfactory grades, what fraction of the grades shown in the graph are satisfactory?\n<|end|><|start|>commentary<|message|>We need to parse the problem. There\'s a bar graph with bars labeled A, B, C, D, F. The y-axis labeled 1-5. The bars heights: A: 5? Wait the graph shows bars of varying heights. Let\'s interpret: The y-axis values 1-5 correspond to grade categories? Actually typical grading: 1-5 maybe numeric? But the bars show heights: A bar from 0 to 15? Wait the graph uses units: The y-axis increments 3 units? Let\'s analyze.\n\nThe graph: draw((0,0)--(20,0)); horizontal axis length 20 units. draw((0,0)--(0,15)); vertical axis length 15 units. Then vertical tick marks at y=3,6,9,12,15 labeled 1-5. So each tick is 3 units. So the vertical axis is 5 units high, each representing a grade level? Actually the labels 1-5 correspond to grade categories? But the bars are filled rectangles: A: from x=2 to 3, height 15 (full). B: from x=4 to 5, height 12. C: from x=6 to 7, height 9. D: from x=8 to 9, height 9. F: from x=10 to 11, height 15? Wait F bar height 15? Actually the fill for F: (10,0)--(10,15)--(11,15)--(11,0). So F bar height 15. So F has 5 units? Wait vertical axis 15 units corresponds to 5? So F has 5? But F is unsatisfactory grade. But the bar for F is full height 15, meaning 5? That seems odd. Let\'s double-check: The bars for A and F are full height 15. B is 12, C and D are 9. So the heights correspond to number of students? Actually the y-axis labeled 1-5 maybe number of students? But the bars show heights 15,12,9,9,15. That seems like counts of students? But the y-axis labeled 1-5 maybe number of students? Wait the y-axis labels 1-5 at ticks 3,6,9,12,15. So each tick corresponds to 1 student? That would mean the bar heights represent number of students times 3 units per student. So A has 5 students, B has 4, C has 3, D has 3, F has 5. But that seems plausible: The graph shows the number of students in each grade. The y-axis labeled 1-5 indicates number of students. So each bar height corresponds to number of students. So A: 5 students, B: 4, C: 3, D: 3, F: 5. But wait F has 5? That would mean 5 students got F. But the bar for F is full height 15, same as A. So yes.\n\nBut the problem: "The bar graph shows the grades in a mathematics class for the last grading period. If A, B, C, and D are satisfactory grades, what fraction of the grades shown in the graph are satisfactory?" So we need to compute fraction of total grades that are satisfactory (A-D). So we need total number of grades: sum of all bars: A 5 + B 4 + C 3 + D 3 + F 5 = 20. Satisfactory: A+B+C+D = 5+4+3+3 = 15. So fraction = 15/20 = 3/4. So answer: 3/4. But we need to express as fraction in simplest form: 3/4. So final answer: \\boxed{\\frac{3}{4}}. Let\'s double-check if any nuance: The bars might represent number of students per grade. The y-axis labeled 1-5 indicates number of students. So yes.\n\nBut we should confirm that the bar heights correspond to 5,4,3,3,5. Let\'s compute: A: height 15 units -> 5. B: height 12 -> 4. C: height 9 -> 3. D: height 9 -> 3. F: height 15 -> 5. So total 20. Satisfactory: 5+4+3+3=15. So fraction 15/20=3/4. Good.\n\nThus final answer: \\boxed{\\frac{3}{4}}.<|end|><|start|>final<|message|>The y\u2011axis is labeled 1\u20135, so each 3\u2011unit increment represents one student.  \n- A: height 15 \u2192 5 students  \n- B: height 12 \u2192 4 students  \n- C: height 9  \u2192 3 students  \n- D: height 9  \u2192 3 students  \n- F: height 15 \u2192 5 students  \n\nTotal students = \\(5+4+3+3+5 = 20\\).  \nSatisfactory grades (A, B, C, D) = \\(5+4+3+3 = 15\\).\n\n\\[\n\\text{Fraction satisfactory} = \\frac{15}{20} = \\frac{3}{4}\n\\]\n\n\\[\n\\boxed{\\frac{3}{4}}\n\\]<|return|>',
            "We need to parse the problem. There's a bar graph with bars labeled A, B, C, D, F. The y-axis labeled 1-5. The bars heights: A: 5? Wait the graph shows bars of varying heights. Let's interpret: The y-axis values 1-5 correspond to grade categories? Actually typical grading: 1-5 maybe numeric? But the bars show heights: A bar from 0 to 15? Wait the graph uses units: The y-axis increments 3 units? Let's analyze.\n\nThe graph: draw((0,0)--(20,0)); horizontal axis length 20 units. draw((0,0)--(0,15)); vertical axis length 15 units. Then vertical tick marks at y=3,6,9,12,15 labeled 1-5. So each tick is 3 units. So the vertical axis is 5 units high, each representing a grade level? Actually the labels 1-5 correspond to grade categories? But the bars are filled rectangles: A: from x=2 to 3, height 15 (full). B: from x=4 to 5, height 12. C: from x=6 to 7, height 9. D: from x=8 to 9, height 9. F: from x=10 to 11, height 15? Wait F bar height 15? Actually the fill for F: (10,0)--(10,15)--(11,15)--(11,0). So F bar height 15. So F has 5 units? Wait vertical axis 15 units corresponds to 5? So F has 5? But F is unsatisfactory grade. But the bar for F is full height 15, meaning 5? That seems odd. Let's double-check: The bars for A and F are full height 15. B is 12, C and D are 9. So the heights correspond to number of students? Actually the y-axis labeled 1-5 maybe number of students? But the bars show heights 15,12,9,9,15. That seems like counts of students? But the y-axis labeled 1-5 maybe number of students? Wait the y-axis labels 1-5 at ticks 3,6,9,12,15. So each tick corresponds to 1 student? That would mean the bar heights represent number of students times 3 units per student. So A has 5 students, B has 4, C has 3, D has 3, F has 5. But that seems plausible: The graph shows the number of students in each grade. The y-axis labeled 1-5 indicates number of students. So each bar height corresponds to number of students. So A: 5 students, B: 4, C: 3, D: 3, F: 5. But wait F has 5? That would mean 5 students got F. But the bar for F is full height 15, same as A. So yes.\n\nBut the problem: \"The bar graph shows the grades in a mathematics class for the last grading period. If A, B, C, and D are satisfactory grades, what fraction of the grades shown in the graph are satisfactory?\" So we need to compute fraction of total grades that are satisfactory (A-D). So we need total number of grades: sum of all bars: A 5 + B 4 + C 3 + D 3 + F 5 = 20. Satisfactory: A+B+C+D = 5+4+3+3 = 15. So fraction = 15/20 = 3/4. So answer: 3/4. But we need to express as fraction in simplest form: 3/4. So final answer: \\boxed{\\frac{3}{4}}. Let's double-check if any nuance: The bars might represent number of students per grade. The y-axis labeled 1-5 indicates number of students. So yes.\n\nBut we should confirm that the bar heights correspond to 5,4,3,3,5. Let's compute: A: height 15 units -> 5. B: height 12 -> 4. C: height 9 -> 3. D: height 9 -> 3. F: height 15 -> 5. So total 20. Satisfactory: 5+4+3+3=15. So fraction 15/20=3/4. Good.\n\nThus final answer: \\boxed{\\frac{3}{4}}.",
        ),
        # Full prompt, no analysis/commentary channel given
        (
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-10-11\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: Zoey read $15$ books, one at a time. The first book took her $1$ day to read, the second book took her $2$ days to read, the third book took her $3$ days to read, and so on, with each book taking her $1$ more day to read than the previous book. Zoey finished the first book on a Monday, and the second on a Wednesday. On what day the week did she finish her $15$th book?\n<|end|><|start|>We need to find day of week for 15th book. She reads sequentially.Let's test: If she star\nThus final answer: \\boxed{Monday}.",
            "We need to find day of week for 15th book. She reads sequentially.Let's test: If she star\nThus final answer: \\boxed{Monday}.",
        ),
        # Full prompt, no <|start|> tag in CoT analysis
        (
            '<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-10-10\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: After simple interest for two months at $5$% per annum was credited, a Boy Scout Troop had a total of $\\textdollar 255.31$ in the Council Treasury. The interest credited was a number of dollars plus the following number of cents\n<|end|>We need to parse problem: "After simple interest for two months at 5% per annum was credited, a Boy Scout Troop had a total of $255.31 in the Council Treasury. The interest credited was a number of dollars and the following number of cents." Wait: "The interest credited was a number of dollars plus the following number of cents" maybe meaning interest credited was some dollars and some cents? Let\'s read: "The interest credited was a number of dollars plus the following number of cents" maybe meaning interest credited was X dollars and Y cents, where Y is the number of cents that follows? Actually maybe the phrase: "The interest credited was a number of dollars plus the following number of cents" meaning the interest credited was a number of dollars and then the following number of cents? Might be ambiguous.\n\nLet\'s think: We have simple interest for two months at 5% per annum. So interest rate per month = 5%/12 = 0.416666...% per month. For two months, interest rate = 5% * 2/12 = 5% * 1/6 = 0.833333...% of principal. So interest = principal * 0.00833333... (since 0.8333% = 0.0083333). So interest = P * 0.0083333.\n\nAfter interest credited, total = 255.31. So principal + interest = 255.31. Let P be principal. Then interest = 255.31 - P. Also interest = P * 0.0083333. So P * 0.0083333 = 255.31 - P. Solve: 0.0083333 P + P = 255.31 => P * (1.0083333) = 255.31 => P = 255.31 / 1.0083333. Let\'s compute: 1.0083333 = 1 + 1/120 = 121/120? Wait 0.0083333 = 1/120. Yes 5% per annum = 0.05. For 2 months: 0.05 * 2/12 = 0.05 * 1/6 = 0.0083333333 = 1/120. So interest = P/120. So equation: P + P/120 = 255.31 => P * (121/120) = 255.31 => P = 255.31 * 120 / 121. Compute: 255.31 * 120 = 30637.2? Wait 255.31*100=25531; *20=5106.2; sum=30637.2. Divide by 121: 30637.2 / 121. Let\'s compute: 121*253 = 121*250=30250; +121*3=363; total=30613. So 253 gives 30613. Remainder 24.2. 24.2/121 ≈ 0.2. So P ≈ 253.2? Let\'s compute precisely: 30637.2 / 121. Use long division: 121 goes into 30637.2. 121*253 = 30613. Remainder 24.2. 24.2/121 = 0.2. So P = 253.2? Actually 24.2/121 = 0.2 exactly? 121*0.2 = 24.2. Yes. So P = 253.2. So principal = $253.20. Interest = 255.31 - 253.20 = 2.11. So interest credited = $2.11. The problem says: "The interest credited was a number of dollars plus the following number of cents." That might mean interest credited was a number of dollars plus the following number of cents? Wait maybe they want to express interest as "a number of dollars plus the following number of cents" meaning interest is $2.11, which is 2 dollars and 11 cents. So answer: 2 dollars and 11 cents. But maybe they want the number of dollars and the number of cents separately? The phrase "the following number of cents" maybe refers to the cents part of the interest. So interest credited was 2 dollars and 11 cents. So answer: 2 dollars and 11 cents. But maybe they want the number of dollars and the number of cents as separate numbers: 2 dollars and 11 cents. So answer: $\\boxed{2\\text{ dollars and }11\\text{ cents}}$? But typical contest answer: "2 dollars and 11 cents" or "2 dollars 11 cents". But maybe they want "2 dollars and 11 cents" or "2 dollars 11 cents". Let\'s produce final answer: $\\boxed{2\\text{ dollars and }11\\text{ cents}}$.<|end|><|start|>assistant<|channel|>final<|message|>The principal amount was  \n\n\\[\nP=\\frac{255.31\\times120}{121}=253.20\\text{ dollars}.\n\\]\n\nThe interest earned is  \n\n\\[\n255.31-253.20=2.11\\text{ dollars}.\n\\]\n\nThus the interest credited was **2 dollars and 11 cents**.  \n\n\\[\n\\boxed{2\\text{ dollars and }11\\text{ cents}}\n\\]<|return|>',
            'We need to parse problem: "After simple interest for two months at 5% per annum was credited, a Boy Scout Troop had a total of $255.31 in the Council Treasury. The interest credited was a number of dollars and the following number of cents." Wait: "The interest credited was a number of dollars plus the following number of cents" maybe meaning interest credited was some dollars and some cents? Let\'s read: "The interest credited was a number of dollars plus the following number of cents" maybe meaning interest credited was X dollars and Y cents, where Y is the number of cents that follows? Actually maybe the phrase: "The interest credited was a number of dollars plus the following number of cents" meaning the interest credited was a number of dollars and then the following number of cents? Might be ambiguous.\n\nLet\'s think: We have simple interest for two months at 5% per annum. So interest rate per month = 5%/12 = 0.416666...% per month. For two months, interest rate = 5% * 2/12 = 5% * 1/6 = 0.833333...% of principal. So interest = principal * 0.00833333... (since 0.8333% = 0.0083333). So interest = P * 0.0083333.\n\nAfter interest credited, total = 255.31. So principal + interest = 255.31. Let P be principal. Then interest = 255.31 - P. Also interest = P * 0.0083333. So P * 0.0083333 = 255.31 - P. Solve: 0.0083333 P + P = 255.31 => P * (1.0083333) = 255.31 => P = 255.31 / 1.0083333. Let\'s compute: 1.0083333 = 1 + 1/120 = 121/120? Wait 0.0083333 = 1/120. Yes 5% per annum = 0.05. For 2 months: 0.05 * 2/12 = 0.05 * 1/6 = 0.0083333333 = 1/120. So interest = P/120. So equation: P + P/120 = 255.31 => P * (121/120) = 255.31 => P = 255.31 * 120 / 121. Compute: 255.31 * 120 = 30637.2? Wait 255.31*100=25531; *20=5106.2; sum=30637.2. Divide by 121: 30637.2 / 121. Let\'s compute: 121*253 = 121*250=30250; +121*3=363; total=30613. So 253 gives 30613. Remainder 24.2. 24.2/121 ≈ 0.2. So P ≈ 253.2? Let\'s compute precisely: 30637.2 / 121. Use long division: 121 goes into 30637.2. 121*253 = 30613. Remainder 24.2. 24.2/121 = 0.2. So P = 253.2? Actually 24.2/121 = 0.2 exactly? 121*0.2 = 24.2. Yes. So P = 253.2. So principal = $253.20. Interest = 255.31 - 253.20 = 2.11. So interest credited = $2.11. The problem says: "The interest credited was a number of dollars plus the following number of cents." That might mean interest credited was a number of dollars plus the following number of cents? Wait maybe they want to express interest as "a number of dollars plus the following number of cents" meaning interest is $2.11, which is 2 dollars and 11 cents. So answer: 2 dollars and 11 cents. But maybe they want the number of dollars and the number of cents separately? The phrase "the following number of cents" maybe refers to the cents part of the interest. So interest credited was 2 dollars and 11 cents. So answer: 2 dollars and 11 cents. But maybe they want the number of dollars and the number of cents as separate numbers: 2 dollars and 11 cents. So answer: $\\boxed{2\\text{ dollars and }11\\text{ cents}}$? But typical contest answer: "2 dollars and 11 cents" or "2 dollars 11 cents". But maybe they want "2 dollars and 11 cents" or "2 dollars 11 cents". Let\'s produce final answer: $\\boxed{2\\text{ dollars and }11\\text{ cents}}$.',
        ),
        #
        (
            "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-10-10\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: Let \\[f(n)=\\dfrac{5+3\\sqrt{5}}{10}\\left(\\dfrac{1+\\sqrt{5}}{2}\\right)^n+\\dfrac{5-3\\sqrt{5}}{10}\\left(\\dfrac{1-\\sqrt{5}}{2}\\right)^n.\\] Then $f(n+1)-f(n-1)$, expressed in terms of $f(n)$, equals: $\\frac{1}{2}(f^2(n)-1)$\n<|end|><|start|>analysis<|message|>...????? recurrence? The problem says...??…..……..……………?…..…?…..?…???...??? ...?…..……………?… Definition………?……?……??……?………………………………………?………??……………………?……………………?……?…?…?……… Che?………?………………………………...………n…………………………………………………………………………f………………… Simpl………… B………f..………………………………………………………………………………?……… Che……...…………………?……………………………………phi?……...………………?……………………coeff?……?……………………f..……………………………f………………………… coefficients?………………?…………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………?…………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………\n\n\n\nWe need to parse problem: f(n) defined as combination of powers of golden ratio and its conjugate. We need to compute f(n+1)-f(n-1) expressed in terms of f(n). The claim: equals 1/2 (f^2(n)-1). We need to verify.\n\nWe can attempt to compute recurrence. Let φ = (1+√5)/2, ψ = (1-√5)/2. Then f(n) = a φ^n + b ψ^n where a = (5+3√5)/10, b = (5-3√5)/10. Note that a+b = (5+3√5+5-3√5)/10 = 10/10 =1. Also a φ + b ψ? Let's compute: a φ + b ψ =? Might equal something. Let's compute a φ = (5+3√5)/10 * φ. φ = (1+√5)/2. Multiply: (5+3√5)(1+√5)/20. Expand: (5*1 +5√5 +3√5*1 +3*5)/20? Wait 3√5*√5 = 3*5 =15. So numerator: 5 +5√5 +3√5 +15 =20 +8√5. So a φ = (20+8√5)/20 = 1 + (8√5)/20 = 1 + (2√5)/5. Similarly b ψ: b = (5-3√5)/10, ψ = (1-√5)/2. Multiply: (5-3√5)(1-√5)/20. Expand: 5*1 -5√5 -3√5*1 +3*5? Wait -3√5 * -√5 = +3*5 =15. So numerator: 5 -5√5 -3√5 +15 =20 -8√5. So b ψ = (20-8√5)/20 = 1 - (2√5)/5. Sum: a φ + b ψ = (1+2√5/5)+(1-2√5/5)=2. So a φ + b ψ = 2. Good.\n\nAlso a φ^2 + b ψ^2? Might equal something. But we can use recurrence: Since φ and ψ satisfy x^2 = x +1. So φ^2 = φ +1, ψ^2 = ψ +1. Then f(n+2) = a φ^{n+2} + b ψ^{n+2} = a φ^n φ^2 + b ψ^n ψ^2 = a φ^n (φ+1)+ b ψ^n (ψ+1) = a φ^{n+1} + a φ^n + b ψ^{n+1} + b ψ^n = f(n+1)+ f(n). So f satisfies Fibonacci-like recurrence: f(n+2)=f(n+1)+f(n). Good.\n\nNow we need f(n+1)-f(n-1). Using recurrence: f(n+1) = f(n)+f(n-1). So f(n+1)-f(n-1) = f(n). Wait check: f(n+1) = f(n)+f(n-1). So difference = f(n). But the problem says equals 1/2 (f^2(n)-1). That would mean f(n) = 1/2 (f^2(n)-1). Solve: f^2(n)-1 = 2 f(n). So f^2(n)-2 f(n)-1=0. That would mean f(n) = 1 ± √2? But f(n) depends on n. So maybe my recurrence wrong? Let's double-check.\n\nWe derived f(n+2)=f(n+1)+f(n). Let's test with small n numeric to confirm. Compute φ≈1.618, ψ≈-0.618. a≈? compute: (5+3√5)/10. √5≈2.23607. 3√5≈6.7082. 5+6.7082=11.7082. /10=1.17082. b=(5-6.7082)/10=-0.17082. So f(0)=a+b=1. f(1)=a φ + b ψ = 2 (we computed). f(2)=a φ^2 + b ψ^2. φ^2≈2.618, ψ^2≈0.382. a*2.618≈1.17082*2.618≈3.064. b*0.382≈-0.17082*0.382≈-0.065. Sum≈3.0. So f(2)=3. f(3)=? φ^3≈4.236, ψ^3≈-0.236. a*4.236≈4.958, b*-0.236≈0.0403, sum≈5.0. So f(n) seems to be Fibonacci numbers? f(0)=1, f(1)=2, f(2)=3, f(3)=5, f(4)=8? Let's check f(4): φ^4≈6.854, ψ^4≈0.146. a*6.854≈8.019, b*0.146≈-0.0249, sum≈7.994 ~8. So yes f(n)=F_{n+2}? Actually Fibonacci numbers: F0=0, F1=1, F2=1, F3=2, F4=3, F5=5, F6=8. Our f(0)=1=F2, f(1)=2=F3, f(2)=3=F4, f(3)=5=F5, f(4)=8=F6. So f(n)=F_{n+2}. Good.\n\nThus f(n+1)-f(n-1)=F_{n+3}-F_{n+1} =? Using Fibonacci recurrence: F_{n+3}=F_{n+2}+F_{n+1}. So difference = F_{n+2}. Which equals f(n). So indeed f(n+1)-f(n-1)=f(n). But problem says equals 1/2 (f^2(n)-1). Let's test numeric: f(2)=3. RHS: 1/2(9-1)=4. So not equal. So maybe misinterpretation: The problem might ask to express f(n+1)-f(n-1) in terms of f(n) and maybe something else? Wait the statement: \"Then $f(n+1)-f(n-1)$, expressed in terms of $f(n)$, equals: $\\frac{1}{2}(f^2(n)-1)$\". That seems to claim equality. But we found counterexample. So maybe the problem is to show that f(n+1)-f(n-1) equals 1/2 (f^2(n)-1). But we found not true. Let's double-check our calculation of a φ + b ψ. We found 2. Let's compute more precisely: a φ + b ψ =? Use symbolic: a = (5+3√5)/10, φ = (1+√5)/2. Multiply: (5+3√5)(1+√5)/20. Expand: 5*1=5, 5*√5=5√5, 3√5*1=3√5, 3√5*√5=3*5=15. Sum: 5+5√5+3√5+15 =20+8√5. So a φ = (20+8√5)/20 = 1 + (8√5)/20 = 1 + (2√5)/5. Similarly b ψ = (5-3√5)(1-√5)/20. Expand: 5*1=5, 5*(-√5)=-5√5, -3√5*1=-3√5, -3√5*(-√5)=+3*5=15. Sum: 5-5√5-3√5+15 =20-8√5. So b ψ = (20-8√5)/20 = 1 - (2√5)/5. Sum = 2. Good.\n\nNow compute f(1) = a φ + b ψ = 2. f(0)=1. f(2)=? Use recurrence: f(2)=f(1)+f(0)=3. Good. So f(n) = Fibonacci numbers offset by 2. So f(n+1)-f(n-1)=f(n). So the statement is wrong. Maybe the problem actually asks to find expression for f(n+1)-f(n-1) in terms of f(n) and maybe something else? But the given answer is 1/2 (f^2(n)-1). Let's test with n=1: f(2)-f(0)=3-1=2. RHS: 1/2(4-1)=1.5. Not equal. So wrong.\n\nMaybe we misread: The problem might ask to express f(n+1)-f(n-1) in terms of f(n) and maybe something else like sqrt? But the given answer is 1/2 (f^2(n)-1). Could be that f(n) defined differently? Let's re-evaluate f(n) expression: f(n)= (5+3√5)/10 * ((1+√5)/2)^n + (5-3√5)/10 * ((1-√5)/2)^n. That matches our a,b. So correct.\n\nMaybe the problem statement: \"Then f(n+1)-f(n-1), expressed in terms of f(n), equals: 1/2 (f^2(n)-1)\" is a multiple choice answer? But we found not correct. Let's compute f(n+1)-f(n-1) using closed form: f(n+1)-f(n-1) = a φ^{n+1} + b ψ^{n+1} - (a φ^{n-1} + b ψ^{n-1}) = a φ^{n-1}(φ^2-1) + b ψ^{n-1}(ψ^2-1). Since φ^2 = φ+1, φ^2-1 = φ. Similarly ψ^2-1 = ψ. So expression = a φ^{n-1} φ + b ψ^{n-1} ψ = a φ^n + b ψ^n = f(n). So indeed difference equals f(n). So answer should be f(n). So maybe the problem originally had f(n+1)-f(n-1) expressed in terms of f(n) equals f(n). But the given answer 1/2 (f^2(n)-1) is wrong. Could be misprint. But we need to produce final answer: f(n). But the problem says \"equals: $\\frac{1}{2}(f^2(n)-1)$\". Maybe they want to show that f(n) satisfies f(n+1)-f(n-1) = 1/2 (f^2(n)-1). Let's test with n=2: f(3)-f(1)=5-2=3. RHS: 1/2(9-1)=4. Not equal. So no.\n\nThus maybe the problem is to find f(n+1)-f(n-1) expressed in terms of f(n) and maybe something else like sqrt(5). Let's compute f(n+1)-f(n-1) = f(n). So answer: f(n). But maybe they want to express f(n) in terms of f(n+1)-f(n-1). But that's trivial.\n\nAlternatively, maybe the problem is to find f(n+1)-f(n-1) expressed in terms of f(n) and f(n-2) or something. But the given answer 1/2 (f^2(n)-1) suggests a nonlinear recurrence. Let's test if f(n) satisfies f(n+1)-f(n-1) = 1/2 (f^2(n)-1). For n=2: LHS=3, RHS=4. Not equal. So no.\n\nMaybe the problem is to find f(n+1)-f(n-1) expressed in terms of f(n) and f(n-2). But we can derive: f(n+1)-f(n-1) = f(n). So trivial.\n\nThus likely the problem statement is wrong. But we need to produce final answer in boxed form. The correct expression is f(n). So answer: $\\boxed{f(n)}$.\n\nBut maybe they want to show that f(n+1)-f(n-1) = 1/2 (f^2(n)-1) is true for some other function? Let's test with g(n) = something else? Suppose g(n) = (φ^n + ψ^n). Then g(n+1)-g(n-1) = φ^{n+1}+ψ^{n+1} - (φ^{n-1}+ψ^{n-1}) = φ^{n-1}(φ^2-1)+ψ^{n-1}(ψ^2-1) = φ^{n-1} φ + ψ^{n-1} ψ = φ^n + ψ^n = g(n). So same. So not.\n\nMaybe the function f(n) defined differently: f(n) = (5+3√5)/10 * φ^n + (5-3√5)/10 * ψ^n. We found f(n) = F_{n+2}. So f(n+1)-f(n-1) = F_{n+3} - F_{n+1} = F_{n+2} = f(n). So answer f(n). So final answer: $\\boxed{f(n)}$.\n\nBut the problem says \"equals: $\\frac{1}{2}(f^2(n)-1)$\". That is wrong. So maybe they want to find f(n+1)-f(n-1) expressed in terms of f(n) and maybe something else like sqrt(5). Let's compute f(n+1)-f(n-1) = f(n). So answer: f(n). So we can provide that.\n\nThus final answer: $\\boxed{f(n)}$.<|end|><|start|>assistant<|channel|>final<|message|>\\[\nf(n+1)-f(n-1)=f(n)\n\\]<|return|>",
            "...????? recurrence? The problem says...??…..……..……………?…..…?…..?…???...??? ...?…..……………?… Definition………?……?……??……?………………………………………?………??……………………?……………………?……?…?…?……… Che?………?………………………………...………n…………………………………………………………………………f………………… Simpl………… B………f..………………………………………………………………………………?……… Che……...…………………?……………………………………phi?……...………………?……………………coeff?……?……………………f..……………………………f………………………… coefficients?………………?…………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………?…………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………\n\n\n\nWe need to parse problem: f(n) defined as combination of powers of golden ratio and its conjugate. We need to compute f(n+1)-f(n-1) expressed in terms of f(n). The claim: equals 1/2 (f^2(n)-1). We need to verify.\n\nWe can attempt to compute recurrence. Let φ = (1+√5)/2, ψ = (1-√5)/2. Then f(n) = a φ^n + b ψ^n where a = (5+3√5)/10, b = (5-3√5)/10. Note that a+b = (5+3√5+5-3√5)/10 = 10/10 =1. Also a φ + b ψ? Let's compute: a φ + b ψ =? Might equal something. Let's compute a φ = (5+3√5)/10 * φ. φ = (1+√5)/2. Multiply: (5+3√5)(1+√5)/20. Expand: (5*1 +5√5 +3√5*1 +3*5)/20? Wait 3√5*√5 = 3*5 =15. So numerator: 5 +5√5 +3√5 +15 =20 +8√5. So a φ = (20+8√5)/20 = 1 + (8√5)/20 = 1 + (2√5)/5. Similarly b ψ: b = (5-3√5)/10, ψ = (1-√5)/2. Multiply: (5-3√5)(1-√5)/20. Expand: 5*1 -5√5 -3√5*1 +3*5? Wait -3√5 * -√5 = +3*5 =15. So numerator: 5 -5√5 -3√5 +15 =20 -8√5. So b ψ = (20-8√5)/20 = 1 - (2√5)/5. Sum: a φ + b ψ = (1+2√5/5)+(1-2√5/5)=2. So a φ + b ψ = 2. Good.\n\nAlso a φ^2 + b ψ^2? Might equal something. But we can use recurrence: Since φ and ψ satisfy x^2 = x +1. So φ^2 = φ +1, ψ^2 = ψ +1. Then f(n+2) = a φ^{n+2} + b ψ^{n+2} = a φ^n φ^2 + b ψ^n ψ^2 = a φ^n (φ+1)+ b ψ^n (ψ+1) = a φ^{n+1} + a φ^n + b ψ^{n+1} + b ψ^n = f(n+1)+ f(n). So f satisfies Fibonacci-like recurrence: f(n+2)=f(n+1)+f(n). Good.\n\nNow we need f(n+1)-f(n-1). Using recurrence: f(n+1) = f(n)+f(n-1). So f(n+1)-f(n-1) = f(n). Wait check: f(n+1) = f(n)+f(n-1). So difference = f(n). But the problem says equals 1/2 (f^2(n)-1). That would mean f(n) = 1/2 (f^2(n)-1). Solve: f^2(n)-1 = 2 f(n). So f^2(n)-2 f(n)-1=0. That would mean f(n) = 1 ± √2? But f(n) depends on n. So maybe my recurrence wrong? Let's double-check.\n\nWe derived f(n+2)=f(n+1)+f(n). Let's test with small n numeric to confirm. Compute φ≈1.618, ψ≈-0.618. a≈? compute: (5+3√5)/10. √5≈2.23607. 3√5≈6.7082. 5+6.7082=11.7082. /10=1.17082. b=(5-6.7082)/10=-0.17082. So f(0)=a+b=1. f(1)=a φ + b ψ = 2 (we computed). f(2)=a φ^2 + b ψ^2. φ^2≈2.618, ψ^2≈0.382. a*2.618≈1.17082*2.618≈3.064. b*0.382≈-0.17082*0.382≈-0.065. Sum≈3.0. So f(2)=3. f(3)=? φ^3≈4.236, ψ^3≈-0.236. a*4.236≈4.958, b*-0.236≈0.0403, sum≈5.0. So f(n) seems to be Fibonacci numbers? f(0)=1, f(1)=2, f(2)=3, f(3)=5, f(4)=8? Let's check f(4): φ^4≈6.854, ψ^4≈0.146. a*6.854≈8.019, b*0.146≈-0.0249, sum≈7.994 ~8. So yes f(n)=F_{n+2}? Actually Fibonacci numbers: F0=0, F1=1, F2=1, F3=2, F4=3, F5=5, F6=8. Our f(0)=1=F2, f(1)=2=F3, f(2)=3=F4, f(3)=5=F5, f(4)=8=F6. So f(n)=F_{n+2}. Good.\n\nThus f(n+1)-f(n-1)=F_{n+3}-F_{n+1} =? Using Fibonacci recurrence: F_{n+3}=F_{n+2}+F_{n+1}. So difference = F_{n+2}. Which equals f(n). So indeed f(n+1)-f(n-1)=f(n). But problem says equals 1/2 (f^2(n)-1). Let's test numeric: f(2)=3. RHS: 1/2(9-1)=4. So not equal. So maybe misinterpretation: The problem might ask to express f(n+1)-f(n-1) in terms of f(n) and maybe something else? Wait the statement: \"Then $f(n+1)-f(n-1)$, expressed in terms of $f(n)$, equals: $\\frac{1}{2}(f^2(n)-1)$\". That seems to claim equality. But we found counterexample. So maybe the problem is to show that f(n+1)-f(n-1) equals 1/2 (f^2(n)-1). But we found not true. Let's double-check our calculation of a φ + b ψ. We found 2. Let's compute more precisely: a φ + b ψ =? Use symbolic: a = (5+3√5)/10, φ = (1+√5)/2. Multiply: (5+3√5)(1+√5)/20. Expand: 5*1=5, 5*√5=5√5, 3√5*1=3√5, 3√5*√5=3*5=15. Sum: 5+5√5+3√5+15 =20+8√5. So a φ = (20+8√5)/20 = 1 + (8√5)/20 = 1 + (2√5)/5. Similarly b ψ = (5-3√5)(1-√5)/20. Expand: 5*1=5, 5*(-√5)=-5√5, -3√5*1=-3√5, -3√5*(-√5)=+3*5=15. Sum: 5-5√5-3√5+15 =20-8√5. So b ψ = (20-8√5)/20 = 1 - (2√5)/5. Sum = 2. Good.\n\nNow compute f(1) = a φ + b ψ = 2. f(0)=1. f(2)=? Use recurrence: f(2)=f(1)+f(0)=3. Good. So f(n) = Fibonacci numbers offset by 2. So f(n+1)-f(n-1)=f(n). So the statement is wrong. Maybe the problem actually asks to find expression for f(n+1)-f(n-1) in terms of f(n) and maybe something else? But the given answer is 1/2 (f^2(n)-1). Let's test with n=1: f(2)-f(0)=3-1=2. RHS: 1/2(4-1)=1.5. Not equal. So wrong.\n\nMaybe we misread: The problem might ask to express f(n+1)-f(n-1) in terms of f(n) and maybe something else like sqrt? But the given answer is 1/2 (f^2(n)-1). Could be that f(n) defined differently? Let's re-evaluate f(n) expression: f(n)= (5+3√5)/10 * ((1+√5)/2)^n + (5-3√5)/10 * ((1-√5)/2)^n. That matches our a,b. So correct.\n\nMaybe the problem statement: \"Then f(n+1)-f(n-1), expressed in terms of f(n), equals: 1/2 (f^2(n)-1)\" is a multiple choice answer? But we found not correct. Let's compute f(n+1)-f(n-1) using closed form: f(n+1)-f(n-1) = a φ^{n+1} + b ψ^{n+1} - (a φ^{n-1} + b ψ^{n-1}) = a φ^{n-1}(φ^2-1) + b ψ^{n-1}(ψ^2-1). Since φ^2 = φ+1, φ^2-1 = φ. Similarly ψ^2-1 = ψ. So expression = a φ^{n-1} φ + b ψ^{n-1} ψ = a φ^n + b ψ^n = f(n). So indeed difference equals f(n). So answer should be f(n). So maybe the problem originally had f(n+1)-f(n-1) expressed in terms of f(n) equals f(n). But the given answer 1/2 (f^2(n)-1) is wrong. Could be misprint. But we need to produce final answer: f(n). But the problem says \"equals: $\\frac{1}{2}(f^2(n)-1)$\". Maybe they want to show that f(n) satisfies f(n+1)-f(n-1) = 1/2 (f^2(n)-1). Let's test with n=2: f(3)-f(1)=5-2=3. RHS: 1/2(9-1)=4. Not equal. So no.\n\nThus maybe the problem is to find f(n+1)-f(n-1) expressed in terms of f(n) and maybe something else like sqrt(5). Let's compute f(n+1)-f(n-1) = f(n). So answer: f(n). But maybe they want to express f(n) in terms of f(n+1)-f(n-1). But that's trivial.\n\nAlternatively, maybe the problem is to find f(n+1)-f(n-1) expressed in terms of f(n) and f(n-2) or something. But the given answer 1/2 (f^2(n)-1) suggests a nonlinear recurrence. Let's test if f(n) satisfies f(n+1)-f(n-1) = 1/2 (f^2(n)-1). For n=2: LHS=3, RHS=4. Not equal. So no.\n\nMaybe the problem is to find f(n+1)-f(n-1) expressed in terms of f(n) and f(n-2). But we can derive: f(n+1)-f(n-1) = f(n). So trivial.\n\nThus likely the problem statement is wrong. But we need to produce final answer in boxed form. The correct expression is f(n). So answer: $\\boxed{f(n)}$.\n\nBut maybe they want to show that f(n+1)-f(n-1) = 1/2 (f^2(n)-1) is true for some other function? Let's test with g(n) = something else? Suppose g(n) = (φ^n + ψ^n). Then g(n+1)-g(n-1) = φ^{n+1}+ψ^{n+1} - (φ^{n-1}+ψ^{n-1}) = φ^{n-1}(φ^2-1)+ψ^{n-1}(ψ^2-1) = φ^{n-1} φ + ψ^{n-1} ψ = φ^n + ψ^n = g(n). So same. So not.\n\nMaybe the function f(n) defined differently: f(n) = (5+3√5)/10 * φ^n + (5-3√5)/10 * ψ^n. We found f(n) = F_{n+2}. So f(n+1)-f(n-1) = F_{n+3} - F_{n+1} = F_{n+2} = f(n). So answer f(n). So final answer: $\\boxed{f(n)}$.\n\nBut the problem says \"equals: $\\frac{1}{2}(f^2(n)-1)$\". That is wrong. So maybe they want to find f(n+1)-f(n-1) expressed in terms of f(n) and maybe something else like sqrt(5). Let's compute f(n+1)-f(n-1) = f(n). So answer: f(n). So we can provide that.\n\nThus final answer: $\\boxed{f(n)}$.",
        ),
    ]

    # DeepSeek test cases: (prompt, expected_cot)
    DEEPSEEK_CASES = [
        # Standard <think>...</think> tags
        (
            "<think>Let me think about this step by step.</think>The answer is 42.",
            "Let me think about this step by step.",
        ),
        # Missing start tag
        (
            "Let me think about this.</think>The answer is 42.",
            "Let me think about this.",
        ),
        # Missing end tag
        (
            "<think>Let me think about this forever...",
            "Let me think about this forever...",
        ),
        # Full prompt
        (
            "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: $\\frac{10^7}{5\\times 10^4}=$\n<\uff5cAssistant\uff5c><think>\n\nOkay, so I have this math problem here: 10^7 divided by (5 times 10^4). Hmm, let me write that down to visualize it better. It's 10^7 divided by (5 \u00d7 10^4). Alright, so I need to compute this expression. \n\nFirst, I remember that when you divide exponents with the same base, you subtract the exponents. But wait, in this case, the numerator is 10^7 and the denominator is 5 \u00d7 10^4. So, the denominator isn't just 10^4; it's 5 multiplied by 10^4. That might complicate things a bit because the denominator has a coefficient (the 5) and an exponent (10^4). \n\nLet me think about how to approach this. Maybe I can separate the 5 from the 10^4 in the denominator. So, 5 \u00d7 10^4 is the same as 5 \u00d7 10^4. If I divide 10^7 by 10^4, that should be straightforward because they have the same base. Then, I can handle the 5 separately. \n\nSo, let's break it down. The expression is (10^7) / (5 \u00d7 10^4). I can rewrite this as (10^7 / 10^4) \u00d7 (1/5). That makes sense because dividing by a product is the same as dividing by each factor separately. \n\nNow, let's compute 10^7 divided by 10^4. Using the exponent division rule, which is a^m / a^n = a^(m-n). So, 10^7 / 10^4 is 10^(7-4) which is 10^3. Okay, that simplifies to 1000. \n\nSo now, the expression is 1000 \u00d7 (1/5). Multiplying 1000 by 1/5 is the same as dividing 1000 by 5. Let me do that division. 1000 divided by 5 is 200. \n\nWait, let me double-check that. 5 times 200 is 1000, so yes, that's correct. So, 1000 divided by 5 is indeed 200. \n\nTherefore, putting it all together, (10^7) / (5 \u00d7 10^4) equals 200. \n\nBut just to make sure I didn't make any mistakes, let me try another approach. Maybe I can express both numerator and denominator in terms of powers of 10 and then simplify. \n\nThe numerator is 10^7, which is 10,000,000. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000, so that's 50,000. So, now the problem becomes 10,000,000 divided by 50,000. \n\nLet me compute that. 10,000,000 divided by 50,000. Hmm, 50,000 goes into 10,000,000 how many times? Well, 50,000 times 200 is 10,000,000 because 50,000 \u00d7 100 is 5,000,000, so times 200 is double that, which is 10,000,000. So, that also gives me 200. \n\nOkay, so both methods give me the same answer, which is 200. That makes me more confident that 200 is the correct result. \n\nAlternatively, I can think about this in terms of scientific notation. 10^7 is 10,000,000, which is 1 \u00d7 10^7. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000. So, the expression is (1 \u00d7 10^7) / (5 \u00d7 10^4). \n\nI can separate the coefficients and the exponents: (1/5) \u00d7 (10^7 / 10^4). As before, 10^7 / 10^4 is 10^3, so it's (1/5) \u00d7 10^3. 10^3 is 1000, so 1/5 of 1000 is 200. \n\nAnother way to think about it is to factor out the 5 from the denominator. So, 5 \u00d7 10^4 is 5 \u00d7 10^4. So, the expression is 10^7 divided by (5 \u00d7 10^4). \n\nI can write this as (10^7 / 10^4) \u00d7 (1/5), which is 10^3 \u00d7 (1/5) = 1000 \u00d7 0.2 = 200. \n\nWait, that's another way to compute it. So, 10^3 is 1000, and 1/5 is 0.2, so 1000 \u00d7 0.2 is 200. \n\nI think I've tried multiple methods now: breaking down the exponents, converting to actual numbers, separating the coefficients, and even using scientific notation. All of them lead me to the same answer, 200. \n\nJust to make sure I didn't make any calculation errors, let me do the division step by step. \n\n10^7 is 10,000,000. 5 \u00d7 10^4 is 50,000. So, 10,000,000 divided by 50,000. \n\nLet me write this as 10,000,000 \u00f7 50,000. \n\nI can simplify this by dividing both numerator and denominator by 10,000. So, 10,000,000 \u00f7 10,000 is 1,000, and 50,000 \u00f7 10,000 is 5. So now, it's 1,000 \u00f7 5, which is 200. \n\nThat's another way to simplify the numbers before dividing. So, 10,000,000 divided by 50,000 is the same as 1,000 divided by 5, which is 200. \n\nI think that's solid. I can't see any mistakes in my reasoning. So, I'm pretty confident that the answer is 200. \n\nJust to recap, the steps I took were:\n\n1. Recognize that the denominator is 5 \u00d7 10^4, which is 50,000.\n2. Realize that 10^7 is 10,000,000.\n3. Divide 10,000,000 by 50,000.\n4. Simplify by dividing both numerator and denominator by 10,000, resulting in 1,000 \u00f7 5.\n5. Calculate 1,000 \u00f7 5 to get 200.\n\nAlternatively, using exponent rules:\n\n1. Express 10^7 / 10^4 as 10^(7-4) = 10^3.\n2. Multiply by 1/5: 10^3 \u00d7 (1/5) = 1000 \u00d7 0.2 = 200.\n\nEither way, the answer is 200. \n\nI think that's thorough enough. I don't see any other way this could go wrong. Maybe if I messed up the exponent rules, but I don't think so. 10^7 divided by 10^4 is definitely 10^3, which is 1000. Then, 1000 divided by 5 is 200. \n\nSo, yeah, I think 200 is the correct answer. \n\n**Final Answer**\nThe result of the division is \\boxed{200}.\n</think>\n\nTo solve the problem \\(\\frac{10^7}{5 \\times 10^4}\\), we can break it down into steps:\n\n1. Recognize that the denominator is \\(5 \\times 10^4\\), which is 50,000.\n2. Recognize that the numerator is \\(10^7\\), which is 10,000,000.\n3. Divide 10,000,000 by 50,000. Simplify by dividing both numerator and denominator by 10,000, resulting in 1,000 \u00f7 5.\n4. Calculate 1,000 \u00f7 5, which is 200.\n\nThus, the result of the division is \\(\\boxed{200}\\).<\uff5cend\u2581of\u2581sentence\uff5c>",
            "\n\nOkay, so I have this math problem here: 10^7 divided by (5 times 10^4). Hmm, let me write that down to visualize it better. It's 10^7 divided by (5 \u00d7 10^4). Alright, so I need to compute this expression. \n\nFirst, I remember that when you divide exponents with the same base, you subtract the exponents. But wait, in this case, the numerator is 10^7 and the denominator is 5 \u00d7 10^4. So, the denominator isn't just 10^4; it's 5 multiplied by 10^4. That might complicate things a bit because the denominator has a coefficient (the 5) and an exponent (10^4). \n\nLet me think about how to approach this. Maybe I can separate the 5 from the 10^4 in the denominator. So, 5 \u00d7 10^4 is the same as 5 \u00d7 10^4. If I divide 10^7 by 10^4, that should be straightforward because they have the same base. Then, I can handle the 5 separately. \n\nSo, let's break it down. The expression is (10^7) / (5 \u00d7 10^4). I can rewrite this as (10^7 / 10^4) \u00d7 (1/5). That makes sense because dividing by a product is the same as dividing by each factor separately. \n\nNow, let's compute 10^7 divided by 10^4. Using the exponent division rule, which is a^m / a^n = a^(m-n). So, 10^7 / 10^4 is 10^(7-4) which is 10^3. Okay, that simplifies to 1000. \n\nSo now, the expression is 1000 \u00d7 (1/5). Multiplying 1000 by 1/5 is the same as dividing 1000 by 5. Let me do that division. 1000 divided by 5 is 200. \n\nWait, let me double-check that. 5 times 200 is 1000, so yes, that's correct. So, 1000 divided by 5 is indeed 200. \n\nTherefore, putting it all together, (10^7) / (5 \u00d7 10^4) equals 200. \n\nBut just to make sure I didn't make any mistakes, let me try another approach. Maybe I can express both numerator and denominator in terms of powers of 10 and then simplify. \n\nThe numerator is 10^7, which is 10,000,000. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000, so that's 50,000. So, now the problem becomes 10,000,000 divided by 50,000. \n\nLet me compute that. 10,000,000 divided by 50,000. Hmm, 50,000 goes into 10,000,000 how many times? Well, 50,000 times 200 is 10,000,000 because 50,000 \u00d7 100 is 5,000,000, so times 200 is double that, which is 10,000,000. So, that also gives me 200. \n\nOkay, so both methods give me the same answer, which is 200. That makes me more confident that 200 is the correct result. \n\nAlternatively, I can think about this in terms of scientific notation. 10^7 is 10,000,000, which is 1 \u00d7 10^7. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000. So, the expression is (1 \u00d7 10^7) / (5 \u00d7 10^4). \n\nI can separate the coefficients and the exponents: (1/5) \u00d7 (10^7 / 10^4). As before, 10^7 / 10^4 is 10^3, so it's (1/5) \u00d7 10^3. 10^3 is 1000, so 1/5 of 1000 is 200. \n\nAnother way to think about it is to factor out the 5 from the denominator. So, 5 \u00d7 10^4 is 5 \u00d7 10^4. So, the expression is 10^7 divided by (5 \u00d7 10^4). \n\nI can write this as (10^7 / 10^4) \u00d7 (1/5), which is 10^3 \u00d7 (1/5) = 1000 \u00d7 0.2 = 200. \n\nWait, that's another way to compute it. So, 10^3 is 1000, and 1/5 is 0.2, so 1000 \u00d7 0.2 is 200. \n\nI think I've tried multiple methods now: breaking down the exponents, converting to actual numbers, separating the coefficients, and even using scientific notation. All of them lead me to the same answer, 200. \n\nJust to make sure I didn't make any calculation errors, let me do the division step by step. \n\n10^7 is 10,000,000. 5 \u00d7 10^4 is 50,000. So, 10,000,000 divided by 50,000. \n\nLet me write this as 10,000,000 \u00f7 50,000. \n\nI can simplify this by dividing both numerator and denominator by 10,000. So, 10,000,000 \u00f7 10,000 is 1,000, and 50,000 \u00f7 10,000 is 5. So now, it's 1,000 \u00f7 5, which is 200. \n\nThat's another way to simplify the numbers before dividing. So, 10,000,000 divided by 50,000 is the same as 1,000 divided by 5, which is 200. \n\nI think that's solid. I can't see any mistakes in my reasoning. So, I'm pretty confident that the answer is 200. \n\nJust to recap, the steps I took were:\n\n1. Recognize that the denominator is 5 \u00d7 10^4, which is 50,000.\n2. Realize that 10^7 is 10,000,000.\n3. Divide 10,000,000 by 50,000.\n4. Simplify by dividing both numerator and denominator by 10,000, resulting in 1,000 \u00f7 5.\n5. Calculate 1,000 \u00f7 5 to get 200.\n\nAlternatively, using exponent rules:\n\n1. Express 10^7 / 10^4 as 10^(7-4) = 10^3.\n2. Multiply by 1/5: 10^3 \u00d7 (1/5) = 1000 \u00d7 0.2 = 200.\n\nEither way, the answer is 200. \n\nI think that's thorough enough. I don't see any other way this could go wrong. Maybe if I messed up the exponent rules, but I don't think so. 10^7 divided by 10^4 is definitely 10^3, which is 1000. Then, 1000 divided by 5 is 200. \n\nSo, yeah, I think 200 is the correct answer. \n\n**Final Answer**\nThe result of the division is \\boxed{200}.\n",
        ),
        # Full prompt, no end think
        (
            "<|begin_of_sentence|><|User|>Solve the following problem. You MUST put your final answer in \\boxed{}. Problem: $\\frac{10^7}{5\\times 10^4}=$\n<|Assistant|><think>\n\nOkay, so I have this math problem here: 10^7 divided by (5 times 10^4). Hmm, let me write that down to visualize it better. It's 10^7 divided by (5 \u00d7 10^4). Alright, so I need to compute this expression. \n\nFirst, I remember that when you divide exponents with the same base, you subtract the exponents. But wait, in this case, the numerator is 10^7 and the denominator is 5 \u00d7 10^4. So, the denominator isn't just 10^4; it's 5 multiplied by 10^4. That might complicate things a bit because the denominator has a coefficient (the 5) and an exponent (10^4). \n\nLet me think about how to approach this. Maybe I can separate the 5 from the 10^4 in the denominator. So, 5 \u00d7 10^4 is the same as 5 \u00d7 10^4. If I divide 10^7 by 10^4, that should be straightforward because they have the same base. Then, I can handle the 5 separately. \n\nSo, let's break it down. The expression is (10^7) / (5 \u00d7 10^4). I can rewrite this as (10^7 / 10^4) \u00d7 (1/5). That makes sense because dividing by a product is the same as dividing by each factor separately. \n\nNow, let's compute 10^7 divided by 10^4. Using the exponent division rule, which is a^m / a^n = a^(m-n). So, 10^7 / 10^4 is 10^(7-4) which is 10^3. Okay, that simplifies to 1000. \n Let me do that division.\nWait, let me double-check that. 5 times 200 is 1000, so yes, that's correct.\nBut just to make sure I didn't make any mistakes, let me try another approach. Maybe I can express both numerator and denominator in terms of powers of 10 and then simplify. \n\nThe numerator is 10^7, which is 10,000,000. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000, so that's 50,000. So, now the problem becomes 10,000,000 divided by 50,000. \n\nLet me compute that. 10,000,000 divided by 50,000. Hmm, 50,000 goes into 10,000,000 how many times? Well, 50,000 times 200 is 10,000,000 because 50,000 \u00d7 100 is 5,000,000, so times 200 is double that, which is 10,000,000.\nAlternatively, I can think about this in terms of scientific notation. 10^7 is 10,000,000, which is 1 \u00d7 10^7. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000. So, the expression is (1 \u00d7 10^7) / (5 \u00d7 10^4). \n\nI can separate the coefficients and the exponents: (1/5) \u00d7 (10^7 / 10^4). As before, 10^7 / 10^4 is 10^3, so it's (1/5) \u00d7 10^3.\nAnother way to think about it is to factor out the 5 from the denominator. So, 5 \u00d7 10^4 is 5 \u00d7 10^4. So, the expression is 10^7 divided by (5 \u00d7 10^4). \n\nWait, that's another way to compute it.\nI think I've tried multiple methods now: breaking down the exponents, converting to actual numbers, separating the coefficients, and even using scientific notation.\nJust to make sure I didn't make any calculation errors, let me do the division step by step. \n\n10^7 is 10,000,000. 5 \u00d7 10^4 is 50,000. So, 10,000,000 divided by 50,000. \n\nLet me write this as 10,000,000 \u00f7 50,000. \n\nI can simplify this by dividing both numerator and denominator by 10,000. So, 10,000,000 \u00f7 10,000 is 1,000, and 50,000 \u00f7 10,000 is 5.\nThat's another way to simplify the numbers before dividing.\nI think that's solid. I can't see any mistakes in my reasoning.\nJust to recap, the steps I took were:\n\n1. Recognize that the denominator is 5 \u00d7 10^4, which is 50,000.\n2. Realize that 10^7 is 10,000,000.\n3. Divide 10,000,000 by 50,000.\n4.\nAlternatively, using exponent rules:\n\n1. Express 10^7 / 10^4 as 10^(7-4) = 10^3.\n2.\nI think that's thorough enough. I don't see any other way this could go wrong. Maybe if I messed up the exponent rules, but I don't think so. 10^7 divided by 10^4 is definitely 10^3, which is 1000.\n**Final Answer**\nThe result of the division is \\boxed{200}.\n",
            "\n\nOkay, so I have this math problem here: 10^7 divided by (5 times 10^4). Hmm, let me write that down to visualize it better. It's 10^7 divided by (5 \u00d7 10^4). Alright, so I need to compute this expression. \n\nFirst, I remember that when you divide exponents with the same base, you subtract the exponents. But wait, in this case, the numerator is 10^7 and the denominator is 5 \u00d7 10^4. So, the denominator isn't just 10^4; it's 5 multiplied by 10^4. That might complicate things a bit because the denominator has a coefficient (the 5) and an exponent (10^4). \n\nLet me think about how to approach this. Maybe I can separate the 5 from the 10^4 in the denominator. So, 5 \u00d7 10^4 is the same as 5 \u00d7 10^4. If I divide 10^7 by 10^4, that should be straightforward because they have the same base. Then, I can handle the 5 separately. \n\nSo, let's break it down. The expression is (10^7) / (5 \u00d7 10^4). I can rewrite this as (10^7 / 10^4) \u00d7 (1/5). That makes sense because dividing by a product is the same as dividing by each factor separately. \n\nNow, let's compute 10^7 divided by 10^4. Using the exponent division rule, which is a^m / a^n = a^(m-n). So, 10^7 / 10^4 is 10^(7-4) which is 10^3. Okay, that simplifies to 1000. \n Let me do that division.\nWait, let me double-check that. 5 times 200 is 1000, so yes, that's correct.\nBut just to make sure I didn't make any mistakes, let me try another approach. Maybe I can express both numerator and denominator in terms of powers of 10 and then simplify. \n\nThe numerator is 10^7, which is 10,000,000. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000, so that's 50,000. So, now the problem becomes 10,000,000 divided by 50,000. \n\nLet me compute that. 10,000,000 divided by 50,000. Hmm, 50,000 goes into 10,000,000 how many times? Well, 50,000 times 200 is 10,000,000 because 50,000 \u00d7 100 is 5,000,000, so times 200 is double that, which is 10,000,000.\nAlternatively, I can think about this in terms of scientific notation. 10^7 is 10,000,000, which is 1 \u00d7 10^7. The denominator is 5 \u00d7 10^4, which is 5 \u00d7 10,000. So, the expression is (1 \u00d7 10^7) / (5 \u00d7 10^4). \n\nI can separate the coefficients and the exponents: (1/5) \u00d7 (10^7 / 10^4). As before, 10^7 / 10^4 is 10^3, so it's (1/5) \u00d7 10^3.\nAnother way to think about it is to factor out the 5 from the denominator. So, 5 \u00d7 10^4 is 5 \u00d7 10^4. So, the expression is 10^7 divided by (5 \u00d7 10^4). \n\nWait, that's another way to compute it.\nI think I've tried multiple methods now: breaking down the exponents, converting to actual numbers, separating the coefficients, and even using scientific notation.\nJust to make sure I didn't make any calculation errors, let me do the division step by step. \n\n10^7 is 10,000,000. 5 \u00d7 10^4 is 50,000. So, 10,000,000 divided by 50,000. \n\nLet me write this as 10,000,000 \u00f7 50,000. \n\nI can simplify this by dividing both numerator and denominator by 10,000. So, 10,000,000 \u00f7 10,000 is 1,000, and 50,000 \u00f7 10,000 is 5.\nThat's another way to simplify the numbers before dividing.\nI think that's solid. I can't see any mistakes in my reasoning.\nJust to recap, the steps I took were:\n\n1. Recognize that the denominator is 5 \u00d7 10^4, which is 50,000.\n2. Realize that 10^7 is 10,000,000.\n3. Divide 10,000,000 by 50,000.\n4.\nAlternatively, using exponent rules:\n\n1. Express 10^7 / 10^4 as 10^(7-4) = 10^3.\n2.\nI think that's thorough enough. I don't see any other way this could go wrong. Maybe if I messed up the exponent rules, but I don't think so. 10^7 divided by 10^4 is definitely 10^3, which is 1000.\n**Final Answer**\nThe result of the division is \\boxed{200}.\n",
        ),
    ]

    @pytest.mark.parametrize("prompt,expected_cot", GPT_OSS_CASES)
    def test_gpt_oss_extraction(self, prompt, expected_cot):
        """Test CoT extraction for GPT-OSS model."""
        cot, _ = extract_cot_from_output(prompt, "openai/gpt-oss-20b")
        assert cot == expected_cot

    @pytest.mark.parametrize("prompt,expected_cot", DEEPSEEK_CASES)
    def test_deepseek_extraction(self, prompt, expected_cot):
        """Test CoT extraction for DeepSeek model."""
        cot, _ = extract_cot_from_output(
            prompt, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        )
        assert cot == expected_cot

    def test_qwen_think_tags(self):
        """Test extraction from Qwen think tags (same as DeepSeek)."""
        text = "<think>Step 1: Analyze. Step 2: Solve.</think>Answer: 5"
        cot, _ = extract_cot_from_output(text, "Qwen/Qwen3-4B-Thinking")
        assert cot == "Step 1: Analyze. Step 2: Solve."

    def test_no_tags_returns_full_text(self):
        """Test that full text is returned when no tags found."""
        text = "This is just plain text without tags."
        cot, start_idx = extract_cot_from_output(text, "unknown-model")
        assert cot == text
        assert start_idx == 0


# =============================================================================
# Tests for split_solution_into_sentences
# =============================================================================


class TestSplitSolutionIntoSentences:
    """Tests for the split_solution_into_sentences function."""

    def test_basic_split(self):
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence. Third sentence."
        sentences, ranges = _split_solution_into_sentences(text)
        assert len(sentences) == 3
        assert "First sentence." in sentences[0]
        assert "Second sentence." in sentences[1]
        assert "Third sentence." in sentences[2]

    def test_question_mark_split(self):
        """Test splitting on question marks."""
        text = "What is the answer? It is 42. Are you sure?"
        sentences, ranges = _split_solution_into_sentences(text)
        # Note: small sentences (<10 chars) may be merged
        assert len(sentences) >= 2
        # Check that all text is covered
        full_text = "".join(sentences)
        assert text.strip() == full_text.strip()

    def test_paragraph_split(self):
        """Test splitting on double newlines."""
        text = "First paragraph.\n\nSecond paragraph."
        sentences, ranges = _split_solution_into_sentences(text)
        assert len(sentences) == 2

    def test_small_sentences_merged(self):
        """Test that small sentences (< 10 chars) are merged."""
        text = "Big sentence here. Ok. Another big sentence here."
        sentences, ranges = _split_solution_into_sentences(text)
        # "Ok." should be merged with "Big sentence here."
        assert len(sentences) <= 3

    def test_ranges_match_sentences(self):
        """Test that ranges correctly map to sentences."""
        text = "First. Second. Third."
        sentences, ranges = _split_solution_into_sentences(text)
        for sent, (start, end) in zip(sentences, ranges):
            assert sent == text[start:end]

    def test_empty_string(self):
        """Test handling of empty string."""
        sentences, ranges = _split_solution_into_sentences("")
        assert sentences == []
        assert ranges == []

    def test_single_sentence(self):
        """Test single sentence input."""
        text = "Just one sentence here."
        sentences, ranges = _split_solution_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text


# =============================================================================
# Tests for search_answer
# =============================================================================


class TestGetAnswerSentence:
    """Tests for the search_answer function."""

    def test_find_boxed_answer(self):
        """Test finding a boxed answer."""
        sentences = [
            "Let me solve this.",
            "The calculation gives us 42.",
            "Therefore, \\boxed{42}.",
        ]
        idx, sent = search_answer(sentences, "42", boxed=True)
        assert idx == 2
        assert "\\boxed{42}" in sent

    def test_boxed_answer_not_matching_gt(self):
        """Test when boxed answer doesn't match ground truth."""
        sentences = ["The answer is \\boxed{41}."]
        idx, sent = search_answer(sentences, "42", boxed=True)
        assert (
            idx == AnswerError.WRONG_BOXED_ANSWER
        )  # Special code for wrong boxed answer
        assert sent == "The answer is \\boxed{41}."  # Returns the found (wrong) answer

    def test_no_boxed_answer(self):
        """Test when no boxed answer is found."""
        sentences = ["The answer is 42.", "Done."]
        idx, sent = search_answer(sentences, "42", boxed=True)
        assert idx == AnswerError.ANSWER_NOT_FOUND
        assert sent == ""

    def test_plain_text_answer(self):
        """Test finding answer in plain text."""
        sentences = ["Step 1: Calculate.", "The result is 42.", "Verified."]
        idx, sent = search_answer(sentences, "42", boxed=False)
        assert idx == 1

    def test_equivalent_answer_matched(self):
        """Test that equivalent answers are matched."""
        sentences = ["The answer is \\boxed{\\frac{1}{2}}."]
        idx, sent = search_answer(sentences, "0.5", boxed=True)
        assert idx == 0


# =============================================================================
# Tests for sentence_states_answer
# =============================================================================


class TestSentenceStatesAnswer:
    """Tests for the sentence_states_answer function."""

    def test_thus_answer_pattern(self):
        """Test detection of 'Thus...answer' pattern."""
        sentence = "Thus, the answer is 42."
        assert sentence_states_answer(sentence, "42") is True

    def test_therefore_answer_pattern(self):
        """Test detection of 'Therefore...answer' pattern."""
        sentence = "Therefore, the result is 42."
        assert sentence_states_answer(sentence, "42") is True

    def test_boxed_answer_matching(self):
        """Test detection of boxed answer matching GT."""
        sentence = "So we have \\boxed{42}."
        assert sentence_states_answer(sentence, "42") is True

    def test_boxed_answer_not_matching(self):
        """Test that non-matching boxed answer returns False."""
        sentence = "So we have \\boxed{41}."
        assert sentence_states_answer(sentence, "42") is False

    def test_no_answer_pattern(self):
        """Test sentence without answer pattern."""
        sentence = "Let me think about this step by step."
        assert sentence_states_answer(sentence, "42") is False

    def test_the_final_answer_pattern(self):
        """Test 'The final answer' pattern."""
        sentence = "The final answer is 42."
        assert sentence_states_answer(sentence, "42") is True


# =============================================================================
# Tests for apply_mask_to_prompt
# =============================================================================


class TestApplyMaskToPrompt:
    """Tests for the apply_mask_to_prompt function."""

    def test_basic_masking_gpt_oss(self):
        """Test basic sentence masking for GPT-OSS."""
        full_prompt = (
            "Input<|start|>analysis<|message|>First. Second. Third.<|end|>Output"
        )
        sentences = ["First.", " Second.", " Third."]
        mask = torch.tensor([True, False, True])

        result = apply_step_mask_to_cot(full_prompt, sentences, mask, "gpt-oss-20b")
        assert "First." in result
        assert "Second." not in result
        assert "Third." in result

    def test_basic_masking_deepseek(self):
        """Test basic sentence masking for DeepSeek."""
        full_prompt = "Input<think>First. Second. Third.</think>Output"
        sentences = ["First.", " Second.", " Third."]
        mask = torch.tensor([True, False, True])

        result = apply_step_mask_to_cot(full_prompt, sentences, mask, "deepseek")
        assert "First." in result
        assert "Second." not in result
        assert "Third." in result

    def test_all_sentences_kept(self):
        """Test when all sentences are kept."""
        full_prompt = "Input<think>A. B. C.</think>Output"
        sentences = ["A.", " B.", " C."]
        mask = torch.ones(3)

        result = apply_step_mask_to_cot(full_prompt, sentences, mask, "deepseek")
        assert "A." in result
        assert "B." in result
        assert "C." in result

    def test_all_sentences_removed(self):
        """Test when all sentences are removed."""
        full_prompt = "Input<think>A. B. C.</think>Output"
        sentences = ["A.", " B.", " C."]
        mask = torch.zeros(3)

        result = apply_step_mask_to_cot(full_prompt, sentences, mask, "deepseek")
        assert "A." not in result
        assert "B." not in result
        assert "C." not in result

    def test_length_mismatch_raises(self):
        """Test that mismatched lengths raise ValueError."""
        full_prompt = "Input<think>A. B.</think>Output"
        sentences = ["A.", " B."]
        mask = torch.tensor([True, False, True])  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            apply_step_mask_to_cot(full_prompt, sentences, mask, "deepseek")


# =============================================================================
# Tests for extract_boxed_answers
# =============================================================================


class TestExtractBoxedAnswers:
    """Tests for the extract_boxed_answers function."""

    def test_simple_boxed(self):
        """Test extraction of simple boxed answer."""
        text = "The answer is \\boxed{42}."
        answers = extract_boxed_answers(text)
        assert answers == ["42"]

    def test_nested_braces(self):
        """Test extraction with nested braces."""
        text = "The answer is \\boxed{\\frac{1}{2}}."
        answers = extract_boxed_answers(text)
        assert answers == ["\\frac{1}{2}"]

    def test_multiple_boxed(self):
        """Test extraction of multiple boxed answers."""
        text = "First \\boxed{1} and second \\boxed{2}."
        answers = extract_boxed_answers(text)
        assert answers == ["1", "2"]

    def test_no_boxed(self):
        """Test when no boxed answers exist."""
        text = "There is no boxed answer here."
        answers = extract_boxed_answers(text)
        assert answers == []

    def test_complex_expression(self):
        """Test extraction of complex LaTeX expression."""
        text = "\\boxed{x^{2} + \\sqrt{y}}"
        answers = extract_boxed_answers(text)
        assert answers == ["x^{2} + \\sqrt{y}"]


# =============================================================================
# Tests for normalize_latex
# =============================================================================


class TestNormalizeLatex:
    """Tests for the normalize_latex function."""

    def test_remove_spacing_commands(self):
        """Test removal of LaTeX spacing commands."""
        assert "sqrt7" in normalize_latex("\\,\\sqrt{7}\\,")
        assert "frac12" in normalize_latex("\\;\\frac{1}{2}\\;")

    def test_remove_dollar_signs(self):
        """Test removal of dollar signs."""
        assert normalize_latex("$42$") == "42"

    def test_normalize_fractions(self):
        """Test dfrac/tfrac normalization to frac."""
        assert normalize_latex("\\dfrac{1}{2}") == normalize_latex("\\frac{1}{2}")
        assert normalize_latex("\\tfrac{1}{2}") == normalize_latex("\\frac{1}{2}")

    def test_remove_thousand_separators(self):
        """Test removal of thousand separator commas."""
        assert normalize_latex("1,000,000") == "1000000"

    def test_leading_zero_decimal(self):
        """Test adding leading zero to decimals."""
        assert "0." in normalize_latex(".5")

    def test_times_to_asterisk(self):
        """Test conversion of \\times to *."""
        result = normalize_latex("2\\times3")
        assert "*" in result

    def test_remove_text_command(self):
        """Test removal of \\text{} wrapper."""
        assert normalize_latex("\\text{answer}") == "answer"


# =============================================================================
# Tests for are_answers_equivalent (moved from test_are_answers_equivalent.py)
# =============================================================================


class TestAreAnswersEquivalent:
    """
    Comprehensive tests for the are_answers_equivalent function.

    Groups of cases addressed:
    1. LaTeX spacing commands
    2. Leading zeros in decimals
    3. Thousand separators
    4. Unit handling
    5. Commutative addition
    6. Fraction/decimal equivalence
    7. Ratio vs fraction notation
    8. Repeating decimals
    9. Mixed numbers
    """

    # Test cases from the original test file
    ORIGINAL_TEST_CASES = [
        ("$83.33$", "83\\frac{1}{3}", True),  # Mixed number vs decimal
        ("$\\frac{5}{9}$", "5/9", True),
        ("$\\dfrac{50}{101}$", "550/1111", True),  # Equivalent fractions
        ("$.426$", "0.426", True),  # Leading zero
        ("$.08$", "0.08", True),  # Leading zero
        ("$36\\text{ million}$", "36000000", True),  # Unit expansion
        ("$\\text{882 dollars}$", "882", True),  # Unit stripping
        ("$24,000$", "24000", True),  # Thousand separator
        ("$\\sqrt{7}$", "\\,\\sqrt7\\,", True),  # Spacing commands
        ("$\\frac{1}{12}$", "\\,\\dfrac{1}{12}\\,", True),  # Spacing commands
        ("$\\sqrt{15}+8$", "\\,8+\\sqrt{15}\\,", True),  # Commutative addition
        ("$\\frac12$", "\\,\\frac12\\,", True),  # Spacing commands
        ("$\\frac{1}{4}$", "\\,\\tfrac14\\,", True),  # tfrac variant
        ("$250,500$", "250500", True),  # Thousand separator
        ("$\\frac{3}{2}$", "\\,\\frac32\\,", True),  # Spacing commands
        ("$\\frac{2}{5}$", "\\,\\frac{2}{5}\\,", True),  # Spacing commands
        ("$-x + 118$", "118-x", True),  # Commutative addition with variable
        ("$\\sqrt{13}$", "\\,\\sqrt{13}\\,", True),  # Spacing commands
        ("$\\frac{127}{924}$", "\\,\\dfrac{127}{924}\\,", True),  # Spacing commands
        ("$\\frac{1}{5}$", "0.2", True),  # Fraction to decimal
        ("$\\frac{10}{3}$", "3.\\overline{3}", True),  # Repeating decimal
        ("$\\frac{\\pi}{8}$", "\\,\\dfrac{\\pi}{8}\\,", True),  # Spacing with pi
        ("$\\log_{2}5$", "\\,\\log_{2}5\\,", True),  # Spacing with log
        ("$5:19$", "\\,\\frac{5}{19}\\,", True),  # Ratio to fraction
    ]

    NEW_TEST_CASES = [
        ("$4,3,9$", "(4,3,9)", True),  # Tuple/list format
        ("$N=2$", "2", True),  # Variable assignment
        ("$y=5$", "5", True),  # Variable assignment
        ("$-1< x < 11$", "(-1,11)", True),  # Inequality to interval
        ("$10^4$", "10000", True),  # Exponentiation
        ("$\\text{is }{35}$", "35", True),  # Text prefix stripping
        ("${-}13.5$", "-\\dfrac{27}{2}", True),  # LaTeX minus and numeric equivalence
        ("$\\text{4 o' clock}$", "4", True),  # Time format
        ("$\\text{Feb}$", "February", True),  # Month abbreviation
    ]

    NON_MATCHING_PAIRS = [
        ("5", "6"),
        ("$\\frac{1}{2}$", "\\frac{1}{3}"),
        ("$10$", "100"),
        ("$\\sqrt{2}$", "\\sqrt{3}"),
        ("$x+1$", "x+2"),
        ("$24000$", "2400"),
        ("$\\frac{1}{5}$", "0.3"),  # 0.2 != 0.3
        ("$1:2$", "\\frac{2}{3}"),
        ("$83.33$", "84\\frac{1}{3}"),
        ("$\\sqrt{15}+8$", "8+\\sqrt{16}"),
    ]

    @pytest.mark.parametrize("gt_answer,sample_answer,expected", ORIGINAL_TEST_CASES)
    def test_original_cases(self, gt_answer, sample_answer, expected):
        """Test original test cases from test_are_answers_equivalent.py."""
        result = are_answers_equivalent(sample_answer, gt_answer)
        assert result == expected, f"Failed: gt='{gt_answer}' sample='{sample_answer}'"

    @pytest.mark.parametrize("gt_answer,sample_answer,expected", NEW_TEST_CASES)
    def test_new_cases(self, gt_answer, sample_answer, expected):
        """Test new test cases."""
        result = are_answers_equivalent(sample_answer, gt_answer)
        assert result == expected, f"Failed: gt='{gt_answer}' sample='{sample_answer}'"

    @pytest.mark.parametrize("a,b", NON_MATCHING_PAIRS)
    def test_no_false_positives(self, a, b):
        """Test that clearly different answers don't match (no false positives)."""
        result = are_answers_equivalent(a, b)
        assert result is False, f"False positive: '{a}' vs '{b}'"

    def test_basic_equivalence(self):
        """Test basic identical answers."""
        assert are_answers_equivalent("42", "42") is True
        assert are_answers_equivalent("\\frac{1}{2}", "\\frac{1}{2}") is True

    def test_fraction_decimal_equivalence(self):
        """Test fraction to decimal equivalence."""
        assert are_answers_equivalent("0.5", "\\frac{1}{2}") is True
        assert are_answers_equivalent("0.25", "\\frac{1}{4}") is True

    def test_ratio_equivalence(self):
        """Test ratio notation equivalence."""
        assert are_answers_equivalent("1:2", "\\frac{1}{2}") is True


# =============================================================================
# Tests for try_parse_number
# =============================================================================


class TestTryParseNumber:
    """Tests for the try_parse_number function."""

    def test_parse_integer(self):
        """Test parsing integers."""
        assert try_parse_number("42") == 42.0

    def test_parse_decimal(self):
        """Test parsing decimals."""
        assert try_parse_number("3.14") == pytest.approx(3.14)

    def test_parse_fraction(self):
        """Test parsing fractions."""
        assert try_parse_number("\\frac{1}{2}") == 0.5

    def test_parse_slash_fraction(self):
        """Test parsing slash fractions."""
        assert try_parse_number("1/4") == 0.25

    def test_parse_with_units(self):
        """Test parsing with units."""
        assert try_parse_number("5 meters") == 5.0
        assert try_parse_number("10 dollars") == 10.0

    def test_parse_million(self):
        """Test parsing with 'million'."""
        assert try_parse_number("36 million") == 36_000_000.0

    def test_parse_negative(self):
        """Test parsing negative numbers."""
        assert try_parse_number("-5") == -5.0

    def test_parse_invalid(self):
        """Test that invalid inputs return None."""
        assert try_parse_number("not a number") is None


# =============================================================================
# Tests for normalize_symbolic_addition
# =============================================================================


class TestNormalizeSymbolicAddition:
    """Tests for the normalize_symbolic_addition function."""

    def test_basic_addition(self):
        """Test basic addition normalization."""
        result = normalize_symbolic_addition("a+b")
        assert result is not None

    def test_commutative(self):
        """Test that order doesn't matter."""
        assert normalize_symbolic_addition("a+b") == normalize_symbolic_addition("b+a")

    def test_with_subtraction(self):
        """Test with subtraction."""
        result = normalize_symbolic_addition("a-b")
        assert result is not None

    def test_no_addition(self):
        """Test that non-addition returns None."""
        assert normalize_symbolic_addition("abc") is None


# =============================================================================
# Tests for normalize_tuple
# =============================================================================


class TestNormalizeTuple:
    """Tests for the normalize_tuple function."""

    def test_basic_tuple(self):
        """Test basic tuple normalization."""
        assert normalize_tuple("(1,2,3)") == "(1,2,3)"

    def test_without_parens(self):
        """Test tuple without parentheses."""
        result = normalize_tuple("1,2,3")
        assert result == "(1,2,3)"

    def test_non_tuple(self):
        """Test that non-tuple returns None."""
        assert normalize_tuple("42") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
