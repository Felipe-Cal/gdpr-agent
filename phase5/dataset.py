"""
Phase 5 — Training dataset generation.

Generates a synthetic GDPR fine-tuning dataset in the Alpaca instruction format:
    {"instruction": "...", "input": "", "output": "..."}

Why Alpaca format?
  It is the most widely supported format for instruction fine-tuning.
  HuggingFace PEFT, Axolotl, and LLaMA-Factory all accept it natively.

Why synthetic data?
  We don't have real user queries (this is a demo project).  We seed the
  dataset from our golden eval pairs (Phase 3) and expand them with paraphrases
  and follow-up questions.  In a production project you would use real queries
  from LangSmith traces, redacted for personal data (GDPR compliance).

Dataset size: ~100 examples — enough to demonstrate the fine-tuning pipeline
without incurring large Vertex AI Training costs.
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed data — the 10 golden eval pairs + expanded variations
# Each tuple is (instruction, output).  We generate paraphrases automatically.
# ---------------------------------------------------------------------------

SEED_PAIRS = [
    (
        "What are the six lawful bases for processing personal data under GDPR?",
        "Under GDPR Article 6(1), the six lawful bases for processing personal data are: "
        "(a) Consent — the data subject has given clear consent; "
        "(b) Contract — processing is necessary to perform a contract with the data subject; "
        "(c) Legal obligation — processing is required by EU or member state law; "
        "(d) Vital interests — processing is necessary to protect someone's life; "
        "(e) Public task — processing is necessary for a task in the public interest or official authority; "
        "(f) Legitimate interests — processing is necessary for the controller's legitimate interests, "
        "unless overridden by the data subject's rights.",
    ),
    (
        "When is a Data Protection Officer mandatory under GDPR?",
        "Under GDPR Article 37, a DPO is mandatory in three cases: "
        "(a) Processing is carried out by a public authority or body (except courts); "
        "(b) Core activities require regular and systematic monitoring of data subjects on a large scale; "
        "(c) Core activities involve large-scale processing of special categories of data (Article 9) "
        "or data relating to criminal convictions (Article 10). "
        "Groups of undertakings may appoint a single DPO.",
    ),
    (
        "What are the breach notification requirements under GDPR Article 33?",
        "Under GDPR Article 33, when a personal data breach occurs, the controller must: "
        "(1) Notify the competent supervisory authority without undue delay and, where feasible, "
        "not later than 72 hours after becoming aware of the breach. "
        "(2) If notification is not made within 72 hours, provide a reasoned justification for the delay. "
        "(3) The notification must include: the nature of the breach, categories and number of data subjects "
        "affected, categories and number of records concerned, name of the DPO, likely consequences, "
        "and measures taken or proposed. "
        "Notification is not required if the breach is unlikely to result in a risk to individuals' rights.",
    ),
    (
        "Explain the right to erasure ('right to be forgotten') under GDPR.",
        "The right to erasure is established in GDPR Article 17. Data subjects can request deletion when: "
        "(a) The data is no longer necessary for the purpose it was collected; "
        "(b) The data subject withdraws consent and there is no other legal basis; "
        "(c) The data subject objects under Article 21 and there are no overriding legitimate grounds; "
        "(d) The data was unlawfully processed; "
        "(e) Erasure is required to comply with a legal obligation. "
        "The right does not apply when processing is necessary for freedom of expression, "
        "compliance with a legal obligation, public health, archiving/research purposes, "
        "or the establishment or defence of legal claims.",
    ),
    (
        "What is the difference between a data controller and a data processor under GDPR?",
        "Under GDPR Article 4: "
        "A 'controller' is a natural or legal person, authority, agency, or other body that, alone or jointly, "
        "determines the purposes and means of the processing of personal data. "
        "A 'processor' is a natural or legal person, authority, agency, or other body that processes "
        "personal data on behalf of the controller. "
        "Key practical difference: controllers decide WHY and HOW data is processed; "
        "processors act only on the controller's documented instructions. "
        "Controllers bear primary GDPR responsibility; processors have more limited obligations "
        "but must have a Data Processing Agreement (Article 28) in place with the controller.",
    ),
    (
        "What constitutes valid consent under GDPR?",
        "Valid consent under GDPR Articles 4(11) and 7 must be: "
        "Freely given — no power imbalance or conditioning of a service on consent; "
        "Specific — granular, for each distinct purpose; "
        "Informed — the data subject must know who is processing, for what purpose, and their rights; "
        "Unambiguous — requires a clear affirmative action (pre-ticked boxes are invalid). "
        "Additional requirements: consent must be as easy to withdraw as to give; "
        "the controller must be able to demonstrate that consent was given (accountability); "
        "for children under 16, parental consent is required (member states may lower to 13).",
    ),
    (
        "When is a Data Protection Impact Assessment (DPIA) required?",
        "Under GDPR Article 35, a DPIA is required before processing that is likely to result in "
        "a high risk to individuals' rights and freedoms. It is mandatory for: "
        "(a) Systematic and extensive profiling with significant effects on individuals; "
        "(b) Large-scale processing of special categories of data (Article 9) or criminal data (Article 10); "
        "(c) Systematic monitoring of publicly accessible areas on a large scale (e.g., CCTV). "
        "Supervisory authorities publish lists of processing operations that always require a DPIA. "
        "If the DPIA reveals a high residual risk that the controller cannot mitigate, "
        "the controller must consult the supervisory authority before starting processing (Article 36).",
    ),
    (
        "What information must be provided when collecting data directly from individuals under GDPR?",
        "Under GDPR Article 13, at the time of collection the controller must provide: "
        "Identity and contact details of the controller (and DPO if applicable); "
        "Purposes and legal basis for processing; "
        "Legitimate interests pursued (if that is the legal basis); "
        "Recipients or categories of recipients; "
        "Details of any transfers to third countries and the safeguards in place; "
        "Retention period or criteria used to determine it; "
        "Data subject rights: access (Art.15), rectification (Art.16), erasure (Art.17), "
        "restriction (Art.18), portability (Art.20), objection (Art.21); "
        "Right to withdraw consent at any time (if consent is the legal basis); "
        "Right to lodge a complaint with a supervisory authority; "
        "Whether providing data is a statutory/contractual requirement and consequences of not providing it; "
        "Existence of automated decision-making including profiling (Art.22).",
    ),
    (
        "What are the special categories of personal data under GDPR Article 9?",
        "GDPR Article 9 defines special categories as personal data revealing or consisting of: "
        "Racial or ethnic origin; Political opinions; Religious or philosophical beliefs; "
        "Trade union membership; Genetic data; Biometric data processed for unique identification; "
        "Health data; Data concerning a person's sex life or sexual orientation. "
        "Processing special categories is prohibited by default. Exceptions include: "
        "explicit consent; employment/social security law obligations; vital interests; "
        "non-profit body activities; data manifestly made public by the subject; "
        "legal claims; substantial public interest; preventive medicine/occupational medicine; "
        "public health; archiving/research/statistics in the public interest.",
    ),
    (
        "Explain the principle of data protection by design and by default under GDPR.",
        "GDPR Article 25 requires controllers to implement data protection from the outset: "
        "By design: implement appropriate technical and organisational measures (e.g., pseudonymisation, "
        "encryption, data minimisation) at the time of determining the means of processing "
        "and at the time of processing itself. "
        "By default: ensure that, by default, only personal data necessary for each specific purpose "
        "is processed — covering the amount collected, extent of processing, storage period, "
        "and accessibility. In particular, data must not be made available to unlimited numbers of people "
        "without the individual's intervention. "
        "Practical examples: switching privacy settings to most restrictive by default, "
        "collecting only mandatory fields, auto-deleting data after retention period.",
    ),
]

# Instruction paraphrase templates — generates variety in the training data
_PARAPHRASE_TEMPLATES = [
    "Explain {topic} under the GDPR.",
    "What does GDPR say about {topic}?",
    "As a GDPR compliance officer, how would you describe {topic}?",
    "Provide a detailed legal explanation of {topic} according to the GDPR.",
    "Summarise the GDPR requirements regarding {topic}.",
    "A client asks: '{original}' How do you respond?",
]

_TOPICS = [
    "lawful bases for data processing",
    "the mandatory DPO requirement",
    "data breach notification timelines",
    "the right to be forgotten",
    "controller vs processor distinction",
    "consent validity requirements",
    "when DPIAs are required",
    "transparency obligations at collection",
    "special categories of personal data",
    "privacy by design and by default",
]


def _generate_paraphrases(instruction: str, output: str, topic: str, n: int = 5):
    """Generate n paraphrase variants of an instruction-output pair."""
    pairs = [(instruction, output)]  # always include the original
    templates = random.sample(_PARAPHRASE_TEMPLATES, min(n, len(_PARAPHRASE_TEMPLATES)))
    for tmpl in templates:
        new_instruction = tmpl.format(topic=topic, original=instruction)
        if new_instruction != instruction:
            pairs.append((new_instruction, output))
    return pairs


def build_dataset(output_path: str = "data/gdpr_finetune.jsonl", seed: int = 42) -> list[dict]:
    """
    Builds the fine-tuning dataset and writes it as JSONL.

    Returns the list of training examples.
    """
    random.seed(seed)
    examples = []

    for (instruction, output), topic in zip(SEED_PAIRS, _TOPICS):
        pairs = _generate_paraphrases(instruction, output, topic, n=8)
        for inst, out in pairs:
            examples.append({
                "instruction": inst,
                "input": "",        # Alpaca format — empty for direct Q&A
                "output": out,
            })

    # Shuffle so topic order doesn't bias training
    random.shuffle(examples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    return examples


if __name__ == "__main__":
    examples = build_dataset()
    print(f"Generated {len(examples)} training examples → data/gdpr_finetune.jsonl")
