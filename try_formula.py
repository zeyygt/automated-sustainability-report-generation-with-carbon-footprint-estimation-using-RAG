"""
End-to-end demo: upload a formula document + a data spreadsheet, confirm the
system picks up the custom emission factors and uses them in calculations.

Run:
    python try_formula.py
"""

from rag_retrieval.session import RetrievalSession

FORMULA_DOC  = "sample_docs/emission_factors.txt"
DATA_DOC     = "sample_docs/consumption_data.csv"
TEST_DISTRICT = "Kadikoy"

REFERENCE_ELECTRICITY = 0.43   # default in reference_Data.json
REFERENCE_GAS         = 2.1
CUSTOM_ELECTRICITY    = 0.99   # declared in emission_factors.txt
CUSTOM_GAS            = 3.50


def separator(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def main() -> None:
    session = RetrievalSession()

    separator("Building index from two documents")
    print(f"  Formula document : {FORMULA_DOC}")
    print(f"  Data spreadsheet : {DATA_DOC}")
    stats = session.build_index([FORMULA_DOC, DATA_DOC])
    print(f"\n  Parsed {stats.document_count} document(s), "
          f"{stats.chunk_count} chunk(s) in {stats.elapsed_seconds:.2f}s")

    # ── What emission factors were found? ────────────────────────────────────
    separator("Emission factors extracted from documents")
    if session.document_emission_factors:
        for doc_id, factors in session.document_emission_factors.items():
            filename = session.documents[doc_id].filename
            for key, val in factors.items():
                print(f"  [{filename}]  {key}: {val}")
    else:
        print("  (none found — reference defaults will be used)")

    # ── Analyse a district ───────────────────────────────────────────────────
    separator(f"District analysis: {TEST_DISTRICT}")
    results_found = False
    for doc_id, engine in session.data_engines.items():
        if engine is None:
            continue
        result = engine.analyze_district(TEST_DISTRICT)
        if not result:
            continue

        filename = session.documents[doc_id].filename
        print(f"\n  Source file: {filename}")
        print(f"  Electricity factor used : {result['emission_factors_used'].get('electricity')}  "
              f"(source: {result['emission_factors_source'].get('electricity')})")
        print(f"  Natural gas factor used : {result['emission_factors_used'].get('natural_gas')}  "
              f"(source: {result['emission_factors_source'].get('natural_gas')})")
        print(f"\n  electricity_emission : {result['electricity_emission']:.2f} kgCO2")
        print(f"  gas_emission         : {result['gas_emission']:.2f} kgCO2")
        print(f"  total_emission       : {result['total_emission']:.2f} kgCO2")
        results_found = True

    if not results_found:
        print(f"  No structured data found for {TEST_DISTRICT}.")

    # ── Verify the custom factors were actually applied ──────────────────────
    separator("Verification")
    for doc_id, engine in session.data_engines.items():
        if engine is None:
            continue
        result = engine.analyze_district(TEST_DISTRICT)
        if not result:
            continue

        used_elec = result["emission_factors_used"].get("electricity")
        used_gas  = result["emission_factors_used"].get("natural_gas")

        elec_ok = abs((used_elec or 0) - CUSTOM_ELECTRICITY) < 0.001
        gas_ok  = abs((used_gas  or 0) - CUSTOM_GAS)         < 0.001

        if elec_ok and gas_ok:
            print(f"  PASS  Custom factors ({CUSTOM_ELECTRICITY}, {CUSTOM_GAS}) "
                  f"were used instead of reference defaults "
                  f"({REFERENCE_ELECTRICITY}, {REFERENCE_GAS}).")
        else:
            if not elec_ok:
                print(f"  FAIL  electricity factor: expected {CUSTOM_ELECTRICITY}, got {used_elec}")
            if not gas_ok:
                print(f"  FAIL  natural_gas factor: expected {CUSTOM_GAS}, got {used_gas}")
        break
    print()


if __name__ == "__main__":
    main()
