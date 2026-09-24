"""Human decisions are independent of provisional model decisions."""
DECISIONS = ('include', 'exclude', 'uncertain')


def initialize_eligibility(paper):
    for key, value in {
        'eligibility_decision': 'uncertain', 'eligibility_status': 'not_screened',
        'eligibility_reason': 'Full-text screening has not run.',
        'eligibility_evidence': [], 'eligibility_criteria': [],
        'fulltext_status': 'not_requested', 'manual_decision': '',
        'manual_reason': '', 'notes': '',
    }.items():
        paper.setdefault(key, value)
    return paper


def final_decision(paper):
    manual = paper.get('manual_decision')
    return manual if manual in DECISIONS else paper.get('eligibility_decision', 'uncertain')
