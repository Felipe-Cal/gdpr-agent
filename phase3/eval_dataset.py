"""
Phase 3 — Golden evaluation dataset for GDPR Legal Analyst.

A collection of high-quality (question, reference_answer) pairs used for
automated scoring. These cover the most important aspects of GDPR to
provide a representative quality score.
"""

EVAL_DATASET = [
    {
        "question": "What are the six lawful bases for processing personal data under Article 6?",
        "reference": "The six lawful bases under GDPR Article 6(1) are: (a) Consent, (b) Performance of a contract, (c) Compliance with a legal obligation, (d) Protection of vital interests, (e) Performance of a task in the public interest or exercise of official authority, and (f) Legitimate interests (unless overridden by the interests or fundamental rights and freedoms of the data subject).",
    },
    {
        "question": "Under what conditions is a Data Protection Officer (DPO) mandatory?",
        "reference": "According to Article 37, a DPO is mandatory if: (a) processing is by a public authority (except courts), (b) core activities require regular and systematic monitoring of data subjects on a large scale, or (c) core activities consist of large-scale processing of special categories of data or criminal convictions.",
    },
    {
        "question": "What are the notification requirements for a personal data breach to the supervisory authority?",
        "reference": "Under Article 33, the controller must notify the competent supervisory authority without undue delay and, where feasible, not later than 72 hours after becoming aware of the breach, unless the breach is unlikely to result in a risk to the rights and freedoms of natural persons.",
    },
    {
        "question": "What is the 'right to be forgotten' and when does it apply?",
        "reference": "The right to erasure (Article 17) allows data subjects to have their data deleted when: it's no longer necessary for the original purpose, consent is withdrawn (with no other legal basis), they object to processing (with no overriding grounds), processing was unlawful, or for compliance with a legal obligation.",
    },
    {
        "question": "What is the difference between a data controller and a data processor?",
        "reference": "Under Article 4, a 'controller' determines the purposes and means of the processing of personal data, while a 'processor' processes personal data on behalf of the controller.",
    },
    {
        "question": "What are the requirements for valid consent under GDPR?",
        "reference": "Valid consent (Article 4 and 7) must be freely given, specific, informed, and an unambiguous indication of the data subject's wishes. It must be as easy to withdraw as it was to give, and the controller must be able to demonstrate that consent was given.",
    },
    {
        "question": "When is a Data Protection Impact Assessment (DPIA) required?",
        "reference": "Article 35 requires a DPIA for processing likely to result in a high risk to rights and freedoms, particularly: systematic and extensive profiling with significant effects, large-scale processing of special categories/criminal data, or large-scale systematic monitoring of public areas.",
    },
    {
        "question": "What info must be provided when collecting data directly from a data subject?",
        "reference": "Under Article 13, info includes: controller/DPO identity, purposes/legal basis, legitimate interests (if used), recipients, third-country transfers, retention period, rights (access, erasure, etc.), right to withdraw consent, and right to lodge a complaint.",
    },
    {
        "question": "How does GDPR define 'special categories' of personal data?",
        "reference": "Article 9 defines special categories as personal data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, trade union membership, genetic data, biometric data for identification, health data, or data concerning a person's sex life or sexual orientation.",
    },
    {
        "question": "What are the 'data protection by design and by default' principles?",
        "reference": "Article 25 requires controllers to implement technical and organisational measures (like pseudonymisation) to implement principles effectively and protect rights (by design) and ensure that, by default, only necessary data is processed for each specific purpose (by default).",
    },
]
