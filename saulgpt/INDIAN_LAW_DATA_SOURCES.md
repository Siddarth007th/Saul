# Indian Law Data Sources (SaulGPT)

As of **March 10, 2026**, these are the highest-value data sources for a production Indian legal assistant knowledge base.

## Tier 1 (Official and primary)

1. **India Code**  
   URL: https://www.indiacode.nic.in/  
   Use for: Central Acts, amendments, rules, section-level statutory text, effective dates.

2. **eGazette of India**  
   URL: https://egazette.gov.in/  
   Use for: Official notifications, commencement dates, delegated legislation, amendment notifications.

3. **Supreme Court of India (official website)**  
   URL: https://www.sci.gov.in/  
   Use for: Supreme Court judgments, orders, causelists, and official court publications.

4. **High Court official websites (state-wise)**  
   Entry point: https://ecourts.gov.in/ecourts_home/  
   Use for: High Court judgments/orders where official portals provide searchable access.

5. **National Judicial Data Grid (NJDG)**  
   URL: https://njdg.ecourts.gov.in/njdgnew/  
   Use for: Court statistics, pendency/disposal analytics, docket-level trend intelligence.

## Tier 2 (Government data ecosystems)

6. **Open Government Data Platform (India)**  
   URL: https://data.gov.in/  
   Use for: Machine-readable government datasets, including justice/governance-related datasets where available.

7. **Department of Legal Affairs / Legislative Department portals**  
   URL: https://legalaffairs.gov.in/  
   Use for: Policy notes, legal reforms context, official legal administration references.

## Tier 3 (Secondary research and enrichment)

8. **Law Commission reports (official archives)**  
   URL: https://lawcommissionofindia.nic.in/  
   Use for: Reform recommendations, doctrinal analysis, historical context.

9. **Curated commercial/legal research platforms (licensed)**  
   Use for: Headnotes, citators, case treatment signals, editorial summaries.  
   Note: Ensure licensing and citation attribution compliance.

## How to make SaulGPT stronger than a basic RAG bot

1. **Statute graph**
- Store section-level nodes with amendment lineage and effective date ranges.
- Keep old and new criminal codes mapped by date context (pre/post 2024-07-01).

2. **Judgment graph**
- Build citation graph and treatment signals: followed, distinguished, overruled, referred.
- Prefer recent binding precedents by court hierarchy and jurisdiction.

3. **Temporal legality checks**
- Always answer with incident-date awareness.
- Distinguish historical IPC/CrPC references from current BNS/BNSS/BSA framework.

4. **Draft intelligence layer**
- Maintain validated templates by matter type: FIR complaint, legal notice, bail, injunction, recovery suit, writ.
- Add checklist engine: jurisdiction, limitation, evidentiary gaps, relief prayer quality.

5. **Trust and traceability**
- Every answer should show source snippets + statute/case citation metadata.
- Penalize low-confidence retrieval and ask targeted clarifying questions instead of hallucinating.

## Compliance and quality guardrails

- Respect each source's terms and robots policy.
- Log provenance for every chunk (URL, title, date, hash, ingestion timestamp).
- Version your knowledge snapshots and support legal rollback audits.
