# src/athena/output/memo.py
"""Generates a strategic memo by calling the Synthesizer LLM.

The memo is generated in Italian, targeting a lawyer audience.
Uses the same LLM infrastructure as the other agents but with a
dedicated synthesizer system prompt.
"""

import json

from athena.agents.llm import _call_model


SYNTHESIZER_SYSTEM_PROMPT = """\
Sei il consulente strategico di ATHENA. Hai i risultati di {n_runs} simulazioni \
con variazione parametrica di profili giudice e stili di advocacy.

Produci un memo strategico per l'avvocato:

1. SINTESI ESECUTIVA (3-5 frasi) — esito più probabile, range probabilità, raccomandazione
2. ANALISI PER SCENARIO — per profilo giudice: probabilità, strategia ottimale, argomenti chiave
3. ARGOMENTI: RANKING DI EFFICACIA — universali vs polarizzanti vs irrilevanti
4. ANALISI DEL PRECEDENTE — tasso adesione vs distinguishing per profilo
5. RACCOMANDAZIONE STRATEGICA — dominante o condizionale, expected value in EUR
6. RISCHI E CAVEAT — limiti simulazione, fattori non modellati, gaps

Vincoli: scrivi per un avvocato, usa i numeri ma spiega cosa significano, \
non mascherare incertezza. 1500-2500 parole.

NOTA: I confidence intervals sono ampi (N={n_per_cell} per cella). Segnala esplicitamente \
dove i dati sono insufficienti per una raccomandazione forte vs dove il segnale \
è chiaro nonostante il campione ridotto.\
"""


def _build_user_prompt(aggregated: dict, case_data: dict) -> str:
    """Build the user prompt with all aggregated data for the synthesizer."""
    sections = []

    # Case info
    case_info = case_data.get("case", {})
    sections.append("## Dati del caso")
    sections.append(f"```json\n{json.dumps(case_info, indent=2, ensure_ascii=False)}\n```")

    # Probability table
    sections.append("\n## Tabella probabilistica")
    prob_table = aggregated.get("probability_table", {})
    readable_table = {}
    for (judge, style), entry in prob_table.items():
        key = f"{judge} / {style}"
        readable_table[key] = {
            "p_annulment": entry["p_annulment"],
            "p_reclassification": entry["p_reclassification"],
            "p_rejection": entry["p_rejection"],
            "ci_rejection": list(entry["ci_rejection"]),
            "n_runs": entry["n_runs"],
        }
    sections.append(f"```json\n{json.dumps(readable_table, indent=2, ensure_ascii=False)}\n```")

    # Argument effectiveness
    sections.append("\n## Efficacia argomenti")
    sections.append(
        f"```json\n{json.dumps(aggregated.get('argument_effectiveness', {}), indent=2, ensure_ascii=False)}\n```"
    )

    # Precedent analysis
    sections.append("\n## Analisi precedenti")
    sections.append(
        f"```json\n{json.dumps(aggregated.get('precedent_analysis', {}), indent=2, ensure_ascii=False)}\n```"
    )

    # Dominated strategies
    dominated = aggregated.get("dominated_strategies", [])
    sections.append("\n## Strategie dominate")
    if dominated:
        sections.append(f"Strategie dominate individuate: {', '.join(dominated)}")
    else:
        sections.append("Nessuna strategia dominata.")

    # Run stats
    sections.append(f"\n## Statistiche simulazione")
    sections.append(f"Run totali: {aggregated.get('total_runs', 0)}")
    sections.append(f"Run falliti: {aggregated.get('failed_runs', 0)}")

    return "\n".join(sections)


def generate_strategic_memo(aggregated: dict, case_data: dict) -> str:
    """Generate a strategic memo using the Synthesizer LLM.

    Args:
        aggregated: Dict from aggregate_results().
        case_data: The case YAML data (parsed dict).

    Returns:
        LLM-generated strategic memo in Italian.
    """
    n_runs = aggregated.get("total_runs", 0)

    # Estimate n_per_cell from first entry
    n_per_cell = 5  # default
    for entry in aggregated.get("probability_table", {}).values():
        n_per_cell = entry.get("n_runs", 5)
        break

    system_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(
        n_runs=n_runs, n_per_cell=n_per_cell
    )
    user_prompt = _build_user_prompt(aggregated, case_data)

    text, _, _, _ = _call_model(system_prompt, user_prompt, temperature=0.4)
    return text
