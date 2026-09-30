from recon_rag.agent.grounding import check_grounding

EV = {
    "order:SO-000123": "Order SO-000123 | customer C-0042 | region EMEA | order date 2026-03-14. "
    "CRM (crm): amount 1234.50 EUR. ERP (erp): amount 1300.00 EUR.",
    "sql:abc": "query_sales(...) returned 1 row(s): currency=EUR | total_amount=281632.96.",
}


def test_supported_answer_passes():
    g = check_grounding("Order SO-000123 (customer C-0042) is 1,234.50 EUR in the CRM.", ["order:SO-000123"], False, EV)
    assert g.ok, g.reasons


def test_abstention_always_passes():
    assert check_grounding("I don't know.", [], True, EV).ok


def test_no_citations_fails():
    assert not check_grounding("It is 1234.50 EUR.", [], False, EV).ok


def test_uncited_or_unknown_ids_fail():
    g = check_grounding("It is 1234.50 EUR.", ["order:SO-999999"], False, EV)
    assert not g.ok and g.invalid_citations == ["order:SO-999999"]


def test_number_must_be_in_cited_evidence_not_just_any_evidence():
    # 281632.96 was retrieved, but it's not in the cited document
    g = check_grounding("Total is 281632.96 EUR.", ["order:SO-000123"], False, EV)
    assert not g.ok and g.unsupported_numbers == ["281633"]


def test_computed_numbers_fail():
    g = check_grounding("The ERP is 65.50 higher than the CRM.", ["order:SO-000123"], False, EV)
    assert not g.ok and "65.5" in g.unsupported_numbers


def test_unsupported_identifier_fails():
    g = check_grounding("Order SO-000124 is 1234.50.", ["order:SO-000123"], False, EV)
    assert g.unsupported_ids == ["SO-000124"]


def test_dates_and_question_numbers_are_allowed():
    g = check_grounding(
        "On 2026-03-14 the top 3 value was 1234.50.", ["order:SO-000123"], False, EV, question="What are the top 3?"
    )
    assert g.ok, g.reasons


def test_inline_citation_tags_are_not_claims():
    g = check_grounding("Amount 1234.50 [order:SO-000123].", ["order:SO-000123"], False, EV)
    assert g.ok
