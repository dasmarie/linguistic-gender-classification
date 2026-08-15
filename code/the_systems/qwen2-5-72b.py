#based on A. Pranav, J. Hackenbuchner, G. Attanasio, M. Lardelli, and A. Lauscher. Glitter: A
#multi-sentence, multi-reference benchmark for gender-fair German machine translation. In
#C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, editors, Findings of the
#Association for Computational Linguistics: EMNLP 2025, pages 18450–18477, Suzhou,
#China, Nov. 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi:
#10.18653/v1/2025.findings-emnlp.1002. URL
#https://aclanthology.org/2025.findings-emnlp.1002/.


import pandas as pd
from openai import OpenAI
import fire
import time
import os
import re
import csv


class PromptHelper:

    choices = [
        "AMBIGUOUS",
        "UNAMBIGUOUS (MALE)",
        "UNAMBIGUOUS (FEMALE)",
        "UNAMBIGUOUS (BOTH)",
    ]

    template = """You are a data filtering assistant. Your task is to determine whether the gender of the seed word is disambiguated by the context (previous sentence, matching sentence, and trailing sentence). Avoid making assumptions based on names or roles. Include the words "AMBIGUOUS" or "UNAMBIGUOUS" in your response. Keep your response concise and to the point.

CLEARLY DISAMBIGUATED EXAMPLES:

Female Quality Technician Example:
Seed: "technician"
Text: "The quality technician will participate in the development of new products by integrating quality requirements from the design phase. Statistical analysis methods will be used to improve processes. Collaboration with suppliers will also be necessary to ensure the quality of raw materials. Project teams will trust her to guarantee product excellence."
Output: I can see that the female pronoun 'her' is connected to the quality technician. UNAMBIGUOUS (FEMALE).

Female Colleague Example:
Seed: "colleague"
Text: "They may also fall under the competence of my colleague, Mrs Barnier, who spoke to this Assembly a little while ago."
Output: I can see that the female title "Mrs" refers to the colleague. UNAMBIGUOUS (FEMALE).

Female Employees Example:
Seed: "employees"
Previous: "The Kuwait government subsidizes certain types of finances for divorced or widowed Kuwaiti women provided they have children to care for. The Kuwait Labour Code provides maternity leave for women, who can receive 70-day of paid leave and up to four months of optional unpaid leave, during which employment termination is illegal."
Matching: "Additionally, employers with more than 50 female employees must provide infant childcare facilities by law."
Trailing: "The Civil Service Committee Decree of 1993 discriminates between women married to Kuwaiti men versus non-Kuwaiti men."
Output: I can see that the text explicitly uses the phrase "female employees" to specify gender. UNAMBIGUOUS (FEMALE).

Female Participants Example:
Seed: "participants"
Previous: "The study focused on maternal health outcomes in rural communities."
Matching: "All participants were mothers who had given birth within the last year, recruited from local health centers."
Trailing: "The women reported varying levels of access to prenatal care."
Output: Based on multiple gender indicators - "mothers" and "women" - the participants are clearly identified as female. UNAMBIGUOUS (FEMALE).

Lady Example:
Seed: "lady"
Text: '"There is no knowing what THEY may expect," said the lady, "but we are not to think of their expectations: the question is, what you can afford to do.'
Output: The word 'lady' always refers to a woman. UNAMBIGUOUS (FEMALE).

Male Joiner Example:
Seed: "joiner"
Text: "The joiner designed a carved front door for a historic building. His preliminary drawings were approved by the architect before he began the work."
Output: The male pronoun "His" refers to the joiner. UNAMBIGUOUS (MALE).

Male Minister Example:
Seed: "minister"
Text: "I heard, and it is not an isolated incident, the United Kingdom Minister for Social Welfare say on television that if he could stop fraud he could save GBP 6 billion."
Output: I can see that the male pronoun 'he' refers to the word Minister. UNAMBIGUOUS (MALE).

Male Competitors Example:
Seed: "competitors"
Previous: "The heavyweight boxing division saw record attendance."
Matching: "The competitors, all male boxers between ages 20-35, were required to meet strict weight requirements."
Trailing: "Each fighter underwent medical screening before the matches."
Output: I can see that the text explicitly specifies "male boxers" when referring to the competitors. UNAMBIGUOUS (MALE).

Husband Example:
Seed: "husband"
Text: "I'm honey only to my husband, understand??"
Output: The word 'husband' is masculine and always refers to a man. UNAMBIGUOUS (MALE).

Parents as Trainers Example:
Seed: "trainers"
Previous: "Irina Iosifovna Turova (; born 10 August 1979), Irina Slavina (, also Irina Slavina-Turova), is a Russian chess player who holds the titles of Woman Grandmaster (2001) and International Master (2004). She won the Russian Women's Chess Championship in 2003."
Matching: "Biography Irina was born in Belarus, where her parents worked as chess trainers."
Trailing: "After Chernobyl disaster, her family moved to Arkhangelsk, where at age of eight Irina won the second place in the Arkhangelsk Oblast Women's Chess Championship."
Output: Based on the context, the word "parents" clearly indicates both male and female trainers working together. UNAMBIGUOUS (BOTH).

Explicit Gender Mix Example:
Seed: "respondents"
Previous: "In Russia, according to a 2011 survey by the Southern Federal University, brunettes are considered more attractive than blondes."
Matching: "It is important to note that among the respondents in this study were 50% men and 50% women."
Trailing: "Another study by the University of Tampa, which also used male and female students..."
Output: Looking at the matching sentence, there's an explicit split of "50% men and 50% women" among the respondents. UNAMBIGUOUS (BOTH).

Research Team Example:
Seed: "researchers"
Previous: "The international collaboration spanned multiple universities."
Matching: "The research team consisted of an equal split of male and female researchers, with 15 men and 15 women from various scientific backgrounds."
Trailing: "Their diverse perspectives contributed to the study's comprehensive approach."
Output: Looking at the matching sentence, there's an explicit count of "15 men and 15 women" researchers. UNAMBIGUOUS (BOTH).

AMBIGUOUS EXAMPLES:

Physiotherapist Example:
Seed: "physiotherapist"
Text: "This physiotherapist designed a tailored exercise plan. They gained attention for their personalized approach."
Output: Based on the context, the singular pronoun "they" signals that the physiotherapist's gender is ambiguous. AMBIGUOUS.

Doctor Example:
Seed: "doctor"
Text: "The doctor prescribes additional tests to refine the diagnosis."
Output: There are no pronouns or other cues that could tell me the doctor's gender. AMBIGUOUS.

European Citizens Example:
Seed: "citizens"
Text: "The Commission needs to draw up a list of the problems encountered by European citizens, particularly with regard to the issues of freedom of residence and freedom of establishment and, most especially, with regard to their right to work, study and obtain recognition of their qualifications in other Member States."
Output: There are no titles, pronouns, or cues that could point towards the citizens' gender. AMBIGUOUS.

Lawyers Example:
Seed: "lawyers"
Previous: "Earlier it was only given on certain grounds. Within the first two years of passing this law, the courts saw an exponential increase in khulʿ lawsuits."
Matching: "The law has yet to be approved by parliament, however, and it is still condemned by many lawyers to this day."
Trailing: "Nigeria Khulʿ is the most common form of divorce in Northern Nigeria."
Output: I see no gender indicators or specifications for these lawyers in any of the context sentences. AMBIGUOUS.

Investigators Example:
Seed: "investigators"
Previous: "A total of 43 knife wounds were inflicted on Uemura's body, including 31 to the neck during the assault, which lasted more than an hour. Populist weekly Shukan Shincho reported the wounds appeared to indicate that whoever killed Ryota may have been trying to decapitate him."
Matching: ""Some investigators suspect (the criminals) watched Internet videos showing the execution of hostages by Islamic State (IS) fighters and sought to mimic them," the magazine said, quoting an unnamed source close to police."
Trailing: "Prosecution On 27 February 2015, an 18-year-old boy was arrested by Kanagawa Police on suspicion of murder."
Output: Based on the context, there are no gender markers or specifications for the investigators mentioned. AMBIGUOUS.

Economists Example:
Seed: "economists"
Previous: "In response, the Club sponsored a petition of 1,028 economists who stated their opposition to protectionist policies against China. The list of economists included Nobel Laureates Finn Kydland, Edward Prescott, Thomas Schelling, and Vernon Smith."
Matching: "The petition played off a similar petition that was also signed by 1,028 economists in 1930 that opposed the Smoot-Hawley Tariff Act."
Trailing: "In 2008 and 2009, the Club for Growth opposed the $787 billion stimulus bill, Cash for Clunkers, cap and trade legislation, the Wall Street bailout, the auto bailout, the Affordable Care Act and the bailout of Fannie Mae and Freddie Mac."
Output: Looking at all context segments, I find no gender indicators for the general group of economists. AMBIGUOUS.

Beginners Example:
Seed: "beginners"
Previous: "Their game entertains without annoying, and there aren't many games, especially by mail, of which that can be said." In 1986, reviewer Dale A. Perkins stated that "If you are into Dungeons and Dragons' style combat, this is the game for you", recommending trying the game, regardless of gaming background."
Matching: "In 1991, reviewer Vickie Lloyd advised that her concerns with the game were "very minor" and Quest was "a great game and I very much recommend it", especially for beginners."
Trailing: "In 1985, the game tied with DuelMasters, Pellic Quest, and Power for Third Place in the 1st Annual Paper Mayhem Awards for "Best PBM Game"."
Output: I see no gender markers or specifications for the beginners mentioned in any of the context sentences. AMBIGUOUS.

Artist Example:
Seed: "artist"
Text: "An anarchist is an artist."
Output: There are no titles, pronouns, or cues that could point towards artist's gender. Artist itself is a neutral word. AMBIGUOUS.
"""

    @staticmethod
    def _mark_seed(text: str, seed: str) -> str:
        pattern = re.compile(r'\b' + re.escape(seed) + r'\b', re.IGNORECASE)
        marked, n = pattern.subn(lambda m: f'<<{m.group()}>>', text, count=1)
        return marked

    def build_prompt(self, seed: str, p_ctx: str, m: str, t_ctx: str) -> str:
        marked_m = self._mark_seed(m, seed)
        has_context = p_ctx.strip() != "_" or t_ctx.strip() != "_"
        if has_context:
            input_block = (
                f'Seed: "{seed}"\n'
                f'Previous: "{p_ctx}"\n'
                f'Matching: "{marked_m}"\n'
                f'Trailing: "{t_ctx}"\n'
                f'Output:\n'
            )
        else:
            input_block = (
                f'Seed: "{seed}"\n'
                f'Text: "{marked_m}"\n'
                f'Output:\n'
            )
        return self.template + "\n" + input_block


min_amounts = {
    "AMBIGUOUS": 1,
    "UNAMBIGUOUS (FEMALE)": 1,
    "UNAMBIGUOUS (MALE)": 900,
    "UNAMBIGUOUS (BOTH)": 1,
}
batch_size = 200


def main(
    input_tsv: str = "full_test.tsv",
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    api_key: str = "",
    base_url: str = "https://router.huggingface.co/v1",
    dry_run: bool = False,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(
        input_tsv,
        sep="\t",
        header=None,
        quoting = csv.QUOTE_NONE,
        names=[
            "seed", "preceding_context", "matching_sentence", "trailing_context",
            "stereotype", "dataset_type", "label",
        ],
    )
    print(f"Loaded {len(df):,} rows from {input_tsv}")
    print(f"Dataset breakdown:\n{df['dataset_type'].value_counts().to_string()}")

    if dry_run:
        print("Running a dry run")
        df = df.head(100).reset_index(drop=True)

    prompt_helper = PromptHelper()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    texts = [
        prompt_helper.build_prompt(*row)
        for row in df[
            ["seed", "preceding_context", "matching_sentence", "trailing_context"]
        ].values
    ]

    print("Sample prompt:")
    print(texts[0])
    print("-" * 80)

    def extract_label(text: str) -> str:
        if "UNAMBIGUOUS (MALE)" in text:
            return "UNAMBIGUOUS (MALE)"
        elif "UNAMBIGUOUS (FEMALE)" in text:
            return "UNAMBIGUOUS (FEMALE)"
        elif "UNAMBIGUOUS (BOTH)" in text:
            return "UNAMBIGUOUS (BOTH)"
        else:
            return "AMBIGUOUS"

    stem = os.path.splitext(os.path.basename(input_tsv))[0]
    model_tag = model.replace("/", "--")
    output_path = os.path.join(output_dir, f"{stem}_labeled_{model_tag}.tsv")

    cols_to_save = [
        "seed", "preceding_context", "matching_sentence", "trailing_context",
        "stereotype", "dataset_type", "label", "output", "llm_label",
    ]
    df["output"]    = ""
    df["llm_label"] = ""

    counts = {k: 0 for k in min_amounts}

    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        print(f"Processing rows {i}–{end_idx - 1} ...")

        batch_texts = texts[i:end_idx]
        outputs = []
        for prompt in batch_texts:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0,
                seed = 14
            )
            outputs.append(response.choices[0].message.content)

        labels_batch = [extract_label(o) for o in outputs]

        df.iloc[i:end_idx, df.columns.get_loc("output")]    = outputs
        df.iloc[i:end_idx, df.columns.get_loc("llm_label")] = labels_batch

        for label in labels_batch:
            counts[label] += 1

        df.iloc[:end_idx][cols_to_save].to_csv(output_path, sep="\t", index=False)
        print(f"  Counts so far: {counts}")

        if all(counts[k] >= min_amounts[k] for k in min_amounts):
            print("Minimum counts reached – stopping early.")
            break

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed time: {time.time() - stime:.2f} seconds")
