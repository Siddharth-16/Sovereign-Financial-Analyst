from app.prose_evidence import build_evidence_brief, select_evidence_passages


def test_goldman_style_executive_overview_is_preserved():
    content = """
Generic FICC commentary about client activity and market-making conditions.

Executive Overview We generated net earnings of $14.28 billion for 2024, compared with $8.52 billion for 2023. Net revenues were $53.51 billion for 2024, 16% higher than 2023, primarily reflecting higher net revenues in Global Banking & Markets and Asset & Wealth Management. The increase in Global Banking & Markets reflected higher Equities and Investment banking fees.
"""
    brief = build_evidence_brief("MD&A discussion of net income drivers", content)
    assert "$14.28 billion" in brief
    assert "$53.51 billion" in brief
    assert "Global Banking & Markets" in brief
    assert "Asset & Wealth Management" in brief


def test_realized_trend_beats_generic_forward_looking_prose():
    content = """
We expect advertising trends to continue to affect revenues and may put pressure on margins. Device mix and product mix could affect monetization.

Google Search & other revenues increased $23.1 billion from 2023 to 2024, driven in part by growth in advertiser spending. YouTube ads revenues increased $4.6 billion from 2023 to 2024.
"""
    passages = select_evidence_passages("advertising revenue trends in MD&A", content, max_passages=1)
    assert "23.1 billion" in passages[0]
    assert "4.6 billion" in passages[0]
