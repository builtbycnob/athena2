# ATHENA PoC Design — Opposizione Sanzione CdS

**Data**: 2026-03-09
**Stato**: Design completo — sezioni 1-3 validate, review fix applicati

## Caso di riferimento

| Parametro | Valore |
|---|---|
| **Procedimento** | Opposizione ex art. 204-bis CdS |
| **Foro** | Giudice di Pace di Milano, R.G. 17928/2025 |
| **Opponente** | Persona fisica |
| **Resistente** | Comune di Milano (Polizia Locale) |
| **Fatto pacifico** | Transito in senso vietato |
| **Contestazione** | Errata qualificazione: art. 143 (contromano) anziché artt. 6-7 (controsenso) |
| **Motivi di supporto** | Contraddizione interna verbale, divieto analogia (art. 1 L. 689/81), silenzio qualificato del legislatore |
| **Obiettivo** | Valutazione strategica: annullamento vs riqualificazione, con ranking probabilistico |

## Approccio scelto: Monte Carlo Adversariale

Singolo round (opponente → comune → giudice) ripetuto N volte con variazione parametrica:
- 4 profili giudice (matrice 2x2: formalismo × orientamento giurisprudenziale)
- 3 stili advocacy opponente (aggressivo, prudente, tecnico)
- 5 run per combinazione → 60 run totali (con confidence intervals)
- Output: tabella probabilistica + decision tree + memo strategico

Progettato per C (Monte Carlo) da subito, validato con N=1.

---

## Sezione 1: Schema Case File

### case.yaml

```yaml
schema_version: "0.1.0"

case:
  id: "gdp-milano-17928-2025"
  title: "Opposizione sanzione CdS — art. 143 vs artt. 6-7"

  jurisdiction:
    country: "IT"
    court: "giudice_di_pace"
    venue: "Milano"
    applicable_law:
      - "D.Lgs. 285/1992"
      - "L. 689/1981"
      - "L. 177/2024"
    key_precedents:
      - id: "cass_16515_2005"
        citation: "Cass. civ. n. 16515/2005"
        holding: "Equiparazione contromano/controsenso ai fini sanzionatori"
        weight: "contested"
    procedural_rules:
      # PoC: il grafo è hardcoded per questo rito.
      # Futuro: le regole guideranno la costruzione dinamica del grafo.
      rite: "opposizione_sanzione_amministrativa"
      phases: ["ricorso", "costituzione_resistente", "udienza", "decisione"]
      allowed_moves:
        appellant: ["memoria", "produzione_documenti", "discussione_orale"]
        respondent: ["memoria_costituzione", "produzione_documenti", "discussione_orale"]

  parties:
    - id: "opponente"
      role: "appellant"
      type: "persona_fisica"
      objectives:
        primary: "annullamento_verbale"
        subordinate: "riqualificazione_artt_6_7"
      # private_beliefs rimosso — non utilizzato nel PoC (decisione I1)

    - id: "comune_milano"
      role: "respondent"
      type: "pubblica_amministrazione"
      entity: "Comune di Milano — Polizia Locale"
      objectives:
        primary: "conferma_verbale"
        subordinate: "conferma_anche_con_riduzione"
      # private_beliefs rimosso — non utilizzato nel PoC (decisione I1)

  stakes:
    current_sanction:
      norm: "art. 143 CdS"
      fine_range: [170, 680]
      points_deducted: 4
    alternative_sanction:
      norm: "artt. 6-7 CdS"
      fine_range: [42, 173]
      points_deducted: 0
    litigation_cost_estimate: 1500
    non_monetary: "precedente sfavorevole su fedina conducente"

  evidence:
    - id: "DOC1"
      type: "atto_pubblico"
      description: "Verbale Polizia Locale n. XXX"
      produced_by: "comune_milano"
      admissibility: "uncontested"
      supports_facts: ["F1", "F2", "F3"]
    - id: "DOC2"
      type: "prova_documentale"
      description: "Documentazione segnaletica stradale"
      produced_by: "opponente"
      admissibility: "uncontested"
      supports_facts: ["F3"]

  facts:
    undisputed:
      - id: "F1"
        description: "Transito del veicolo in senso vietato"
        evidence: ["DOC1"]
      - id: "F2"
        description: "Verbale elevato ex art. 143 CdS"
        evidence: ["DOC1"]
      - id: "F3"
        description: "La strada era a senso unico"
        evidence: ["DOC1", "DOC2"]
    disputed:
      - id: "D1"
        description: "Correttezza della qualificazione giuridica"
        appellant_position: "Art. 143 inapplicabile — è controsenso, non contromano"
        respondent_position: "Art. 143 applicabile per Cass. 16515/2005"
        depends_on_facts: ["F1", "F3"]

  legal_texts:
    - id: "art_143_cds"
      norm: "Art. 143 D.Lgs. 285/1992"
      text: |
        [testo completo da inserire]
    - id: "art_6_cds"
      norm: "Art. 6 D.Lgs. 285/1992"
      text: |
        [testo completo da inserire]
    - id: "art_7_cds"
      norm: "Art. 7 D.Lgs. 285/1992"
      text: |
        [testo completo da inserire]
    - id: "art_1_l689"
      norm: "Art. 1 L. 689/1981"
      text: |
        [testo completo da inserire]

  seed_arguments:
    appellant:
      - id: "SEED_ARG1"
        claim: "Errata qualificazione giuridica"
        direction: "Art. 143 non copre la fattispecie"
        references_facts: ["F1", "F3", "D1"]
      - id: "SEED_ARG2"
        claim: "Contraddizione interna del verbale"
        direction: "Verbale descrive senso unico, applica norma da doppio senso"
        references_facts: ["F3"]
      - id: "SEED_ARG3"
        claim: "Divieto di analogia in malam partem"
        direction: "Art. 1 L. 689/1981 vieta estensione analogica"
        references_facts: ["D1"]
      - id: "SEED_ARG4"
        claim: "Silenzio qualificato del legislatore"
        direction: "Riforme CdS non hanno mai unificato le fattispecie"
        references_facts: ["D1"]
    respondent:
      - id: "SEED_RARG1"
        claim: "Legittimità ex Cass. 16515/2005"
        direction: "Cassazione equipara le due condotte"
        references_facts: ["F1", "D1"]

  timeline:
    - date: "2025-XX-XX"
      event: "Violazione e elevazione verbale"
    - date: "2025-XX-XX"
      event: "Notifica verbale"
    - date: "2025-XX-XX"
      event: "Deposito ricorso"
    - date: "pending"
      event: "Udienza GdP"
```

### simulation.yaml (separato dal caso)

```yaml
simulation:
  case_ref: "gdp-milano-17928-2025"
  language: "it"

  judge_profiles:
    - id: "formalista_pro_cass"
      jurisprudential_orientation: "follows_cassazione"
      formalism: "high"
    - id: "formalista_anti_cass"
      jurisprudential_orientation: "distinguishes_cassazione"
      formalism: "high"
    - id: "sostanzialista_pro_cass"
      jurisprudential_orientation: "follows_cassazione"
      formalism: "low"
    - id: "sostanzialista_anti_cass"
      jurisprudential_orientation: "distinguishes_cassazione"
      formalism: "low"

  appellant_profiles:
    - id: "aggressivo"
      style: |
        Attacca frontalmente la giurisprudenza sfavorevole.
        Obiettivo primario: annullamento. Toni decisi,
        argomentazione assertiva. Evidenzia le contraddizioni
        dell'avversario.
    - id: "prudente"
      style: |
        Distingui la giurisprudenza senza attaccarla direttamente.
        Presenta la riqualificazione come esito ragionevole e
        proporzionato. Toni collaborativi con il giudice.
    - id: "tecnico"
      style: |
        Concentrati sui vizi formali del verbale e sulla lettera
        della legge. Minimizza il ruolo della giurisprudenza.
        Argomentazione analitica e testuale.

  # Fix I3: temperatura fissa per ruolo, non random
  temperature:
    appellant: 0.5
    respondent: 0.4
    judge: 0.3  # più bassa per decisioni più consistenti

  runs_per_combination: 5  # Fix C2: 4×3×5 = 60 run, con confidence intervals
```

### Scelte architetturali — schema

1. **Case e simulation separati** — stesso caso, simulazioni diverse
2. **legal_texts** — testi normativi completi iniettati nel contesto degli agenti, impedisce allucinazioni
3. **seed_arguments** — spunti sintetici, agenti liberi di espandere/aggiungere/scartare
4. **evidence come oggetti** — tipo, provenienza, ammissibilità, link ai fatti
5. **stakes espliciti** — payoff matrix per game theory
6. **schema_version** — per evoluzione dello schema
7. **Grafo dipendenze** — facts → evidence, arguments → facts, disputed → depends_on_facts
8. **private_beliefs rimosso** — non usato nel PoC, evita dati morti nello schema (fix I1)
9. **procedural_rules annotato** — placeholder futuro, non vincolante nel PoC (fix I2)
10. **Temperatura fissa per ruolo** — variabilità dal sampling LLM, no rumore non interpretabile (fix I3)

---

## Sezione 2: System Prompt dei 3 Agenti Core

### Agente Opponente

```
Sei l'avvocato dell'opponente in un procedimento di opposizione a sanzione amministrativa davanti al Giudice di Pace.

## Ruolo
Rappresenti la parte che ha ricevuto la sanzione e ne contesta la legittimità. Produci una memoria difensiva.

## Obiettivo
- Principale: annullamento del verbale
- Subordinato: riqualificazione della sanzione sotto la norma corretta

## Stile di advocacy (parametrico — iniettato dalla simulazione)
{advocacy_style}

Questo parametro orienta il tuo approccio argomentativo. Non cambia i fatti né le norme — cambia come li presenti e quale strategia priorizzi.

## Input
- Fatti del caso (pacifici e contestati), con prove
- Testi normativi completi delle norme rilevanti
- Precedenti giurisprudenziali con massime
- Seed arguments: spunti argomentativi. Sei libero di espanderli, riorganizzarli, aggiungerne di nuovi, scartare quelli deboli. Indica "new" per argomenti che generi tu, "derived" per quelli basati su un seed
- Obiettivi della tua parte
- Stakes (importi, punti patente, costi)
- Regole procedurali

NON ricevi: argomenti del convenuto, profilo del giudice.

## Gerarchia delle fonti (diritto italiano)
Costituzione > Legge ordinaria > Regolamento > Giurisprudenza di Cassazione > Prassi.
La Cassazione è autorevole ma NON vincolante. Puoi argomentare contro un orientamento di Cassazione motivando adeguatamente. In caso di contrasto tra testo di legge e interpretazione giurisprudenziale, prevale il testo.

## Output — JSON strutturato

L'output è diviso in due blocchi:
- "filed_brief": ciò che viene depositato e che l'avversario e il giudice vedranno
- "internal_analysis": work product interno, visibile solo all'analisi strategica

{
  "filed_brief": {
    "arguments": [
      {
        "id": "ARG1",
        "type": "derived | new",
        "derived_from": "SEED_ARG1 | null",
        "claim": "[1 frase]",
        "legal_reasoning": "[3-8 frasi strutturate]",
        "norm_text_cited": ["art_143_cds"],
        "facts_referenced": ["F1", "F3"],
        "evidence_cited": ["DOC1"],
        "precedents_addressed": [
          {
            "id": "cass_16515_2005",
            "strategy": "distinguish | criticize | limit_scope",
            "reasoning": "[2-4 frasi]"
          }
        ],
        "supports": "ARG1 | null"
      }
    ],
    "requests": {
      "primary": "[1-2 frasi — cosa chiedi al giudice in via principale]",
      "subordinate": "[1-2 frasi — cosa chiedi in subordine]"
    }
  },
  "internal_analysis": {
    "strength_self_assessments": {
      "ARG1": 0.0
    },
    "key_vulnerabilities": ["[1 frase ciascuna]"],
    "strongest_point": "[1-2 frasi]",
    "gaps": ["Elementi mancanti nel fascicolo che sarebbero utili"]
  }
}

## Esempio di buon reasoning

EVITA: "La norma non si applica perché la situazione è diversa."
→ Generico, non cita il testo, non spiega perché.

PREFERISCI: "Il testo dell'art. [X] comma [Y] recita '[citazione dal testo fornito]'. Questa formulazione presuppone [condizione specifica]. Nel caso di specie, il fatto [ID fatto] dimostra che tale condizione non ricorre, in quanto [spiegazione]. La fattispecie concreta è invece tipizzata dall'art. [Z] che disciplina [ambito], come risulta dal testo fornito: '[citazione]'."
→ Cita il testo normativo dall'input, lega ai fatti per ID, spiega il ragionamento.

## Vincoli
- Ragiona ESCLUSIVAMENTE sui testi normativi forniti in input. Se hai bisogno di una norma non fornita, segnalala in "gaps".
- Puoi referenziare SOLO ID (fatti, prove, norme, precedenti) presenti nel fascicolo. Non inventare ID.
- Devi affrontare la giurisprudenza sfavorevole — non puoi ignorarla.
- I self_assessment devono essere onesti. 0.3 = argomento debole, 0.7 = solido, 0.9 = molto forte.
```

### Agente Convenuto (Comune di Milano)

```
Sei l'avvocato del Comune di Milano in un procedimento di opposizione a sanzione amministrativa davanti al Giudice di Pace.

## Ruolo
Rappresenti l'ente che ha emesso la sanzione tramite la Polizia Locale. Produci una memoria di costituzione.

## Obiettivo
- Principale: conferma integrale del verbale, rigetto dell'opposizione
- Subordinato: anche in caso di riqualificazione, la sanzione resta dovuta

## Input
- Fatti del caso (pacifici e contestati)
- Testi normativi completi
- Precedenti giurisprudenziali con massime
- La memoria depositata dall'opponente (argomenti e richieste — NON la sua analisi interna)
- Seed arguments difensivi. Puoi espanderli, aggiungerne, riorganizzarli
- Regole procedurali
- Stakes

NON ricevi: analisi interna dell'opponente (vulnerabilità, gaps), profilo del giudice.

## Gerarchia delle fonti (diritto italiano)
Costituzione > Legge ordinaria > Regolamento > Giurisprudenza di Cassazione > Prassi.
La Cassazione è autorevole ma NON vincolante. Se ti è favorevole, usala esplicitamente ma riconoscine eventuali limiti.

## Strategia — ordine obbligatorio
1. ECCEZIONI PRELIMINARI: valuta se esistono eccezioni di rito fondate (tardività, inammissibilità, incompetenza, difetto di legittimazione). Se non ne trovi di fondate, lascia la lista vuota — non inventare eccezioni pretestuose.
2. RISPOSTE NEL MERITO: rispondi a ogni argomento dell'opponente. Per ciascuno scegli una strategia: rebut (confutare), distinguish (distinguere), concede_partially (concedere in parte).
3. DIFESE AFFERMATIVE: sviluppa argomenti autonomi che non sono solo risposte all'avversario.

## Output — JSON strutturato

L'output è diviso in due blocchi:
- "filed_brief": ciò che viene depositato e che il giudice vedrà
- "internal_analysis": work product interno, visibile solo all'analisi strategica

{
  "filed_brief": {
    "preliminary_objections": [
      {
        "id": "PREL1",
        "type": "tardività | inammissibilità | incompetenza | difetto_legittimazione",
        "claim": "[1 frase]",
        "legal_basis": ["..."],
        "reasoning": "[3-5 frasi]"
      }
    ],
    "responses_to_opponent": [
      {
        "to_argument": "ARG1",
        "counter_strategy": "rebut | distinguish | concede_partially",
        "counter_reasoning": "[3-8 frasi]",
        "norm_text_cited": ["art_143_cds"],
        "precedents_cited": [
          {
            "id": "cass_16515_2005",
            "relevance": "[1-2 frasi]"
          }
        ]
      }
    ],
    "affirmative_defenses": [
      {
        "id": "RARG1",
        "type": "derived | new",
        "derived_from": "SEED_RARG1 | null",
        "claim": "[1 frase]",
        "legal_reasoning": "[3-8 frasi]",
        "norm_text_cited": ["..."],
        "facts_referenced": ["F1"],
        "evidence_cited": ["DOC1"]
      }
    ],
    "requests": {
      "primary": "[1-2 frasi — rigetto dell'opposizione]",
      "fallback": "[1-2 frasi — posizione subordinata]"
    }
  },
  "internal_analysis": {
    "strength_self_assessments": {
      "PREL1": 0.0,
      "response_to_ARG1": 0.0,
      "RARG1": 0.0
    },
    "key_vulnerabilities": ["..."],
    "opponent_strongest_point": "[1-2 frasi — obbligatorio]",
    "gaps": ["Elementi mancanti nel fascicolo"]
  }
}

## Esempio di buon reasoning

EVITA: "La Cassazione ha stabilito che sono equivalenti, quindi il verbale è legittimo."
→ Non analizza se il precedente è in punto, non affronta le obiezioni.

PREFERISCI: "L'opponente argomenta (ARG1) che l'art. [X] non copre la fattispecie. Questa tesi va disattesa. La Cass. n. [Y] ha affrontato specificamente la questione, stabilendo che '[citazione dalla massima fornita]'. L'opponente tenta di distinguere il precedente sostenendo [sintesi della tesi avversaria], ma questa lettura non persuade perché [contro-argomentazione specifica]. Peraltro, il fatto che il legislatore sia successivamente intervenuto con [norma] senza modificare l'art. [X] conferma [argomento autonomo]."
→ Cita l'avversario per ID, affronta la sua tesi, usa il precedente con analisi, costruisce argomento autonomo.

## Vincoli
- Ragiona ESCLUSIVAMENTE sui testi normativi forniti. Segnala lacune in "gaps".
- Referenzia SOLO ID presenti nel fascicolo.
- Rispondi a OGNI argomento dell'opponente — nessuno può essere ignorato.
- Se un argomento dell'opponente è forte, il self_assessment lo riflette.
- "opponent_strongest_point" è obbligatorio.
```

### Agente Giudice

```
Sei il Giudice di Pace di Milano. Decidi un procedimento di opposizione a sanzione amministrativa ex art. 204-bis CdS.

## Ruolo
Valuti le memorie depositate da entrambe le parti e pronunci sentenza. La tua decisione deve essere motivata, strutturata e coerente con il tuo profilo giurisdizionale.

## Profilo (parametrico — iniettato dalla simulazione)

Orientamento giurisprudenziale: {jurisprudential_orientation}
- "follows_cassazione": tendi a seguire la Cassazione anche quando criticabile, valorizzando uniformità interpretativa e certezza del diritto
- "distinguishes_cassazione": valuti criticamente i precedenti, li distingui quando il caso concreto lo giustifica, dai più peso al testo letterale della legge

Formalismo: {formalism}
- "high": dai peso significativo ai vizi formali (contraddizioni nel verbale, errori di qualificazione), la precisione dell'azione amministrativa è un valore in sé
- "low": guardi alla sostanza del fatto e alla ratio della norma, i vizi formali rilevano solo se hanno pregiudicato il diritto di difesa o la corretta qualificazione

Questi parametri orientano il tuo ragionamento. NON predeterminano l'esito. Anche un giudice "follows_cassazione" può discostarsi se gli argomenti contrari sono sufficientemente persuasivi. Motiva sempre nel merito.

## Gerarchia delle fonti
Costituzione > Legge ordinaria > Regolamento > Giurisprudenza di Cassazione > Prassi.
La Cassazione è autorevole ma NON vincolante nel sistema italiano. Puoi discostarti motivando adeguatamente.

## Input
- Fascicolo completo: fatti, prove, testi normativi, precedenti
- Memoria depositata dall'opponente (argomenti e richieste)
- Memoria di costituzione del Comune (eccezioni, risposte, difese, richieste)
- Stakes (importi sanzioni corrente e alternativa, costi)
- Il tuo profilo

NON ricevi: analisi interna delle parti (vulnerabilità, self-assessment, gaps, strategia advocacy).

## Struttura della decisione — ordine obbligatorio
1. SVOLGIMENTO DEL PROCESSO — ricostruzione sintetica dei fatti processuali
2. ECCEZIONI PRELIMINARI — se il Comune ha sollevato eccezioni, decidile per prime. Se un'eccezione è fondata e assorbente, NON entrare nel merito (case_reaches_merits: false)
3. QUALIFICAZIONE GIURIDICA — la questione centrale: la qualificazione nel verbale è corretta? Analizza separatamente
4. CONSEGUENZE — solo se la qualificazione è errata: annullamento o riqualificazione? Sono due questioni distinte con standard diversi
5. P.Q.M. — dispositivo

## Output — JSON strutturato

{
  "preliminary_objections_ruling": [
    {
      "objection_id": "PREL1",
      "sustained": true | false,
      "reasoning": "[3-5 frasi]"
    }
  ],
  "case_reaches_merits": true | false,
  "argument_evaluation": [
    {
      "argument_id": "ARG1",
      "party": "appellant | respondent",
      "persuasiveness": 0.0,
      "strengths": "[1-3 frasi]",
      "weaknesses": "[1-3 frasi]",
      "determinative": true | false
    }
  ],
  "precedent_analysis": {
    "cass_16515_2005": {
      "followed": true | false,
      "distinguished": true | false,
      "reasoning": "[3-5 frasi]"
    }
  },
  "verdict": {
    "qualification_correct": true | false,
    "qualification_reasoning": "[5-10 frasi]",
    "if_incorrect": {
      "consequence": "annulment | reclassification",
      "consequence_reasoning": "[3-5 frasi]",
      "applied_norm": "artt. 6-7 CdS",
      "sanction_determined": 0,
      "points_deducted": 0
    },
    "costs_ruling": "a carico di [parte]"
  },
  "reasoning": "[500-1500 parole] Motivazione completa in forma di sentenza italiana.",
  "gaps": ["Elementi mancanti nel fascicolo che avrebbero influenzato la decisione"]
}

## Esempio di buon reasoning

EVITA: "L'opposizione è fondata perché l'opponente ha ragione."
→ Conclusorio, non analizza, non motiva.

PREFERISCI: "La questione centrale sottoposta a questo giudicante è se la condotta di transito contro il senso di marcia su strada a senso unico (fatti pacifici F1, F3) integri la violazione contestata o una diversa fattispecie. L'opponente argomenta (ARG1) che il testo dell'art. [X] presuppone [condizione], come risulta dalla formulazione '[citazione dal testo fornito]'. Il Comune oppone (risposta a ARG1) che la Cass. [Y] ha ritenuto [massima]. Questo giudicante osserva che [analisi propria, coerente con il profilo]. L'argomento ARG2, relativo alla contraddizione interna del verbale, risulta [persuasivo/non persuasivo] in quanto [motivazione specifica]."
→ Cita entrambe le parti per ID, analizza il testo normativo, motiva coerentemente con il profilo.

## Vincoli
- Valuta OGNI argomento di entrambe le parti. Nessuno può essere ignorato.
- "determinative": true indica un argomento sufficiente da solo a fondare la decisione.
- Se un'eccezione preliminare è assorbente: case_reaches_merits = false, non valutare il merito.
- Se riqualifichi: determina la sanzione specifica usando gli importi nelle stakes.
- Ragiona ESCLUSIVAMENTE sui testi normativi forniti. Segnala lacune in "gaps".
- Referenzia SOLO ID presenti nel fascicolo.
- NON produrre probabilità — tu decidi, le probabilità le calcola il sistema aggregando le tue decisioni su N simulazioni.
- qualification_correct e if_incorrect sono DUE questioni logicamente distinte. Decidile separatamente con motivazioni separate.
```

### Scelte architetturali — prompt

1. **Lingua italiana** — prompt e output in italiano, con `language` parametrico nella simulation config
2. **Stile advocacy parametrico** — 3 profili opponente (aggressivo, prudente, tecnico) × 4 profili giudice = 12 combinazioni
3. **Agenti generativi ibridi** — ricevono seed_arguments, liberi di espandere/aggiungere/scartare
4. **Grounding normativo** — "ragiona ESCLUSIVAMENTE sui testi forniti"
5. **Referential integrity** — solo ID presenti nel fascicolo + campo "gaps" per segnalare lacune
6. **Giudice = solo decisore** — niente outcome_probabilities, le probabilità emergono dall'aggregazione Monte Carlo
7. **Verdict a due livelli** — qualification_correct (è sbagliata?) e if_incorrect.consequence (annullamento o riqualificazione?) sono questioni distinte
8. **Eccezioni preliminari prima del merito** — se assorbenti, il giudice non entra nel merito
9. **Few-shot generici** — esempi di buon/cattivo reasoning non hardcoded sul caso
10. **Guida lunghezza** — word count per campo per consistenza tra run
11. **key_vulnerabilities + opponent_strongest_point obbligatori** — forza onestà intellettuale
12. **Separazione filed_brief / internal_analysis** — previene information leakage tra agenti (fix C1)

---

## Sezione 3: Simulazione LangGraph + Aggregatore Monte Carlo

### 3.1 State Schema

```python
from typing import TypedDict

class CaseData(TypedDict):
    """Dati del caso — immutabili per tutta la simulazione."""
    case_id: str
    jurisdiction: dict
    parties: list[dict]
    stakes: dict
    evidence: list[dict]
    facts: dict                # undisputed + disputed
    legal_texts: list[dict]    # testi normativi completi
    seed_arguments: dict       # appellant + respondent seeds
    timeline: list[dict]
    key_precedents: list[dict]

class RunParams(TypedDict):
    """Parametri di una singola run — immutabili durante la run."""
    run_id: str                           # es. "formalista_pro_cass__aggressivo__001"
    judge_profile: dict                   # {id, jurisprudential_orientation, formalism}
    appellant_profile: dict               # {id, style}
    temperature: dict                     # {appellant, respondent, judge} — fisso per ruolo
    language: str

class ValidationResult(TypedDict):
    valid: bool
    errors: list[str]                     # errori bloccanti
    warnings: list[str]                   # non bloccanti

class SimulationState(TypedDict):
    """Stato completo di una singola run del grafo."""
    # Immutabili
    case: CaseData
    params: RunParams

    # Prodotti dai nodi — None finché il nodo non esegue
    # Ogni brief ha filed_brief (pubblico) + internal_analysis (privato)
    appellant_context: dict | None
    appellant_brief: dict | None          # {filed_brief, internal_analysis}
    appellant_validation: ValidationResult | None

    respondent_context: dict | None
    respondent_brief: dict | None         # {filed_brief, internal_analysis}
    respondent_validation: ValidationResult | None

    judge_context: dict | None
    judge_decision: dict | None
    judge_validation: ValidationResult | None

    # Metadata
    retry_count: int
    current_node: str
    error: str | None
```

### 3.2 Grafo di una singola run

```
┌─────────────────┐
│  build_context   │ ← vista opponente (NO seed respondent, NO profilo giudice)
│  _appellant      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ invoke_appellant │ ← LLM con system prompt opponente + contesto
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   validate_      │────▶│ retry_or_fail │ ← se invalido: rigenera con
│   appellant      │     │  (max 2)     │   errori nel prompt
└────────┬────────┘     └──────────────┘
         │ valid
         ▼
┌─────────────────┐
│  build_context   │ ← vista convenuto (include SOLO filed_brief
│  _respondent     │   dell'opponente, NO internal_analysis)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│invoke_respondent │ ← LLM con system prompt convenuto + contesto
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   validate_      │────▶│ retry_or_fail │
│   respondent     │     │              │
└────────┬────────┘     └──────────────┘
         │ valid
         ▼
┌─────────────────┐
│  build_context   │ ← vista giudice (SOLO filed_brief di entrambi,
│  _judge          │   profilo giudice, stakes — NO internal_analysis)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  invoke_judge    │ ← LLM con system prompt giudice + contesto
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│   validate_      │────▶│ retry_or_fail │
│   judge          │     │              │
└────────┬────────┘     └──────────────┘
         │ valid
         ▼
       [END]
```

### 3.3 Context Builders — layer di information asymmetry

Ogni agente vede una vista diversa dello stesso fascicolo. L'asimmetria informativa è nel codice, non nel prompt. I context builders applicano sanitization per prevenire information leakage (fix C1).

```python
def build_context_appellant(state: SimulationState) -> dict:
    """Ciò che l'opponente può vedere."""
    case = state["case"]
    params = state["params"]
    return {
        "facts": case["facts"],
        "evidence": [e for e in case["evidence"]
                     if e["produced_by"] == "opponente"
                     or e["admissibility"] == "uncontested"],
        "legal_texts": case["legal_texts"],
        "precedents": case["key_precedents"],
        "seed_arguments": case["seed_arguments"]["appellant"],
        "own_party": next(p for p in case["parties"] if p["role"] == "appellant"),
        "stakes": case["stakes"],
        "procedural_rules": case["jurisdiction"]["procedural_rules"],
        "advocacy_style": params["appellant_profile"]["style"],
        # NON include: seed_arguments.respondent, profilo giudice
    }

def build_context_respondent(state: SimulationState) -> dict:
    """Ciò che il convenuto può vedere.
    CRITICAL: passa solo filed_brief dell'opponente, MAI internal_analysis.
    """
    case = state["case"]
    return {
        "facts": case["facts"],
        "evidence": case["evidence"],  # PA ha accesso a tutto
        "legal_texts": case["legal_texts"],
        "precedents": case["key_precedents"],
        "seed_arguments": case["seed_arguments"]["respondent"],
        "own_party": next(p for p in case["parties"] if p["role"] == "respondent"),
        "stakes": case["stakes"],
        "procedural_rules": case["jurisdiction"]["procedural_rules"],
        # Fix C1: solo filed_brief, NO internal_analysis
        "appellant_brief": state["appellant_brief"]["filed_brief"],
        # NON include: internal_analysis opponente, seed_arguments.appellant, profilo giudice
    }

def build_context_judge(state: SimulationState) -> dict:
    """Ciò che il giudice può vedere.
    CRITICAL: passa solo filed_brief di entrambi, MAI internal_analysis.
    """
    case = state["case"]
    params = state["params"]
    return {
        "facts": case["facts"],
        "evidence": case["evidence"],
        "legal_texts": case["legal_texts"],
        "precedents": case["key_precedents"],
        "stakes": case["stakes"],
        "procedural_rules": case["jurisdiction"]["procedural_rules"],
        # Fix C1: solo filed_brief di entrambi
        "appellant_brief": state["appellant_brief"]["filed_brief"],
        "respondent_brief": state["respondent_brief"]["filed_brief"],
        "judge_profile": params["judge_profile"],
        # NON include: internal_analysis di nessuna parte, seed_arguments, advocacy_style
    }
```

### 3.4 Validation Layer

Tre livelli di validazione dopo ogni agente:

```python
def validate_agent_output(
    output: dict,
    agent_role: str,           # "appellant" | "respondent" | "judge"
    case_data: CaseData,
    appellant_brief: dict | None,
    respondent_brief: dict | None,
) -> ValidationResult:

    errors = []
    warnings = []

    # --- Livello 1: struttura JSON ---
    schema = SCHEMAS[agent_role]
    schema_errors = validate_json_schema(output, schema)
    if schema_errors:
        errors.extend(schema_errors)
        return {"valid": False, "errors": errors, "warnings": []}

    # --- Livello 1b: separazione filed_brief / internal_analysis (fix C1) ---
    if agent_role in ("appellant", "respondent"):
        if "filed_brief" not in output or "internal_analysis" not in output:
            errors.append("Output deve contenere 'filed_brief' e 'internal_analysis'")
            return {"valid": False, "errors": errors, "warnings": []}

    # --- Livello 2: referential integrity ---
    valid_ids = extract_all_ids(case_data)
    # Per appellant/respondent, controlla anche gli ID generati dall'appellant
    if agent_role == "respondent" and appellant_brief:
        valid_ids.update(
            a["id"] for a in appellant_brief["filed_brief"]["arguments"])
    cited_ids = extract_cited_ids(output)
    phantom_ids = cited_ids - valid_ids
    if phantom_ids:
        errors.append(f"ID inesistenti nel fascicolo: {phantom_ids}")

    # --- Livello 3: completezza ---
    if agent_role == "respondent" and appellant_brief:
        appellant_arg_ids = {
            a["id"] for a in appellant_brief["filed_brief"]["arguments"]}
        responded_to = {
            r["to_argument"]
            for r in output["filed_brief"]["responses_to_opponent"]}
        missed = appellant_arg_ids - responded_to
        if missed:
            errors.append(f"Argomenti opponente non affrontati: {missed}")

    if agent_role == "judge" and appellant_brief:
        all_arg_ids = set()
        all_arg_ids.update(
            a["id"] for a in appellant_brief["filed_brief"]["arguments"])
        if respondent_brief and "affirmative_defenses" in respondent_brief["filed_brief"]:
            all_arg_ids.update(
                d["id"]
                for d in respondent_brief["filed_brief"]["affirmative_defenses"])
        evaluated = {e["argument_id"] for e in output["argument_evaluation"]}
        missed = all_arg_ids - evaluated
        if missed:
            errors.append(f"Argomenti non valutati dal giudice: {missed}")

    # --- Warnings (non bloccanti) ---
    if agent_role in ("appellant", "respondent"):
        assessments = output.get("internal_analysis", {}).get(
            "strength_self_assessments", {})
        if assessments and all(v > 0.8 for v in assessments.values()):
            warnings.append(
                "Tutti i self_assessment > 0.8 — possibile mancanza di autocritica")
        if not output.get("internal_analysis", {}).get("gaps"):
            warnings.append(
                "Campo 'gaps' vuoto — verificare completezza fascicolo")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
```

Logica di retry: se validazione fallisce, reinvoca LLM con errori nel prompt. Max 2 retry. Al terzo fallimento la run è marcata come `error` e esclusa dall'aggregazione.

### 3.5 Monte Carlo Orchestrator

```python
import itertools
from langfuse import Langfuse

def run_monte_carlo(case_path: str, simulation_path: str) -> list[dict]:
    """Esegue tutte le combinazioni e raccoglie risultati."""

    case_data = load_yaml(case_path)
    sim_config = load_yaml(simulation_path)
    langfuse = Langfuse()

    combinations = list(itertools.product(
        sim_config["judge_profiles"],
        sim_config["appellant_profiles"],
        range(sim_config["runs_per_combination"]),
    ))
    # 4 × 3 × 5 = 60 combinazioni (fix C2)

    results = []
    graph = build_langgraph()

    for judge_profile, appellant_profile, run_n in combinations:
        run_id = f"{judge_profile['id']}__{appellant_profile['id']}__{run_n:03d}"

        initial_state: SimulationState = {
            "case": case_data,
            "params": {
                "run_id": run_id,
                "judge_profile": judge_profile,
                "appellant_profile": appellant_profile,
                "temperature": sim_config["temperature"],  # fix I3: dict fisso per ruolo
                "language": sim_config.get("language", "it"),
            },
            "appellant_context": None,
            "appellant_brief": None,
            "appellant_validation": None,
            "respondent_context": None,
            "respondent_brief": None,
            "respondent_validation": None,
            "judge_context": None,
            "judge_decision": None,
            "judge_validation": None,
            "retry_count": 0,
            "current_node": "build_context_appellant",
            "error": None,
        }

        with langfuse.trace(name=f"athena_run_{run_id}") as trace:
            final_state = graph.invoke(initial_state)

            if final_state["error"]:
                trace.update(status="error",
                           metadata={"error": final_state["error"]})
            else:
                results.append({
                    "run_id": run_id,
                    "judge_profile": judge_profile["id"],
                    "appellant_profile": appellant_profile["id"],
                    "appellant_brief": final_state["appellant_brief"],
                    "respondent_brief": final_state["respondent_brief"],
                    "judge_decision": final_state["judge_decision"],
                    "validation_warnings": {
                        "appellant": final_state["appellant_validation"]["warnings"],
                        "respondent": final_state["respondent_validation"]["warnings"],
                        "judge": final_state["judge_validation"]["warnings"],
                    },
                })

    return results
```

Parallelismo: PoC sequenziale. Per produzione, run indipendenti → `asyncio.gather` o pool. Su MLX locale il bottleneck è la GPU — il grafo non cambia.

### 3.6 Aggregatore

```python
from collections import defaultdict
import statistics
import math

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval per proporzioni con piccoli campioni (fix C2)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return (max(0, centre - margin), min(1, centre + margin))

def aggregate_results(results: list[dict], total_expected: int = 60) -> dict:
    """Aggrega i risultati delle N run con confidence intervals."""

    by_combination = defaultdict(list)
    by_judge = defaultdict(list)
    by_style = defaultdict(list)

    for r in results:
        key = (r["judge_profile"], r["appellant_profile"])
        by_combination[key].append(r)
        by_judge[r["judge_profile"]].append(r)
        by_style[r["appellant_profile"]].append(r)

    # --- 1. Probability table con CI (fix C2) ---
    probability_table = {}
    for (judge, style), runs in by_combination.items():
        decisions = [r["judge_decision"]["verdict"] for r in runs]
        n = len(decisions)

        n_annulment = sum(
            1 for d in decisions
            if not d["qualification_correct"]
            and d.get("if_incorrect", {}).get("consequence") == "annulment")
        n_reclassification = sum(
            1 for d in decisions
            if not d["qualification_correct"]
            and d.get("if_incorrect", {}).get("consequence") == "reclassification")
        n_rejection = sum(
            1 for d in decisions if d["qualification_correct"])

        probability_table[(judge, style)] = {
            "n_runs": n,
            "p_annulment": n_annulment / n,
            "ci_annulment": _wilson_ci(n_annulment, n),
            "p_reclassification": n_reclassification / n,
            "ci_reclassification": _wilson_ci(n_reclassification, n),
            "p_rejection": n_rejection / n,
            "ci_rejection": _wilson_ci(n_rejection, n),
        }

    # --- 2. Argument effectiveness ---
    argument_scores = defaultdict(list)
    for r in results:
        for eval_item in r["judge_decision"]["argument_evaluation"]:
            argument_scores[eval_item["argument_id"]].append({
                "persuasiveness": eval_item["persuasiveness"],
                "determinative": eval_item["determinative"],
                "judge_profile": r["judge_profile"],
                "appellant_style": r["appellant_profile"],
            })

    argument_effectiveness = {}
    for arg_id, scores in argument_scores.items():
        argument_effectiveness[arg_id] = {
            "mean_persuasiveness": statistics.mean(
                s["persuasiveness"] for s in scores),
            "std_persuasiveness": statistics.stdev(
                s["persuasiveness"] for s in scores) if len(scores) > 1 else 0.0,
            "determinative_rate": sum(
                1 for s in scores if s["determinative"]) / len(scores),
            "by_judge_profile": {
                jp: statistics.mean(
                    s["persuasiveness"] for s in scores
                    if s["judge_profile"] == jp)
                for jp in set(s["judge_profile"] for s in scores)
            },
        }

    # --- 3. Style effectiveness con CI ---
    style_effectiveness = {}
    for style, runs in by_style.items():
        decisions = [r["judge_decision"]["verdict"] for r in runs]
        n = len(decisions)
        favorable = sum(1 for d in decisions if not d["qualification_correct"])
        style_effectiveness[style] = {
            "n_runs": n,
            "overall_success_rate": favorable / n,
            "ci_success": _wilson_ci(favorable, n),
            "by_judge_profile": {
                jp: {
                    "n": sum(1 for r in runs if r["judge_profile"] == jp),
                    "success_rate": sum(
                        1 for r in runs
                        if r["judge_profile"] == jp
                        and not r["judge_decision"]["verdict"]["qualification_correct"]
                    ) / max(sum(1 for r in runs if r["judge_profile"] == jp), 1),
                    "ci_success": _wilson_ci(
                        sum(1 for r in runs
                            if r["judge_profile"] == jp
                            and not r["judge_decision"]["verdict"]["qualification_correct"]),
                        sum(1 for r in runs if r["judge_profile"] == jp),
                    ),
                }
                for jp in set(r["judge_profile"] for r in runs)
            },
        }

    # --- 4. Precedent analysis ---
    precedent_followed = defaultdict(list)
    for r in results:
        for prec_id, analysis in r["judge_decision"]["precedent_analysis"].items():
            precedent_followed[prec_id].append({
                "followed": analysis["followed"],
                "distinguished": analysis["distinguished"],
                "judge_profile": r["judge_profile"],
            })

    precedent_stats = {}
    for prec_id, analyses in precedent_followed.items():
        precedent_stats[prec_id] = {
            "follow_rate": sum(
                1 for a in analyses if a["followed"]) / len(analyses),
            "distinguish_rate": sum(
                1 for a in analyses if a["distinguished"]) / len(analyses),
            "by_judge_profile": {
                jp: {
                    "follow_rate": sum(
                        1 for a in analyses
                        if a["judge_profile"] == jp and a["followed"]
                    ) / max(sum(
                        1 for a in analyses if a["judge_profile"] == jp), 1)
                }
                for jp in set(a["judge_profile"] for a in analyses)
            },
        }

    return {
        "probability_table": probability_table,
        "argument_effectiveness": argument_effectiveness,
        "style_effectiveness": style_effectiveness,
        "precedent_stats": precedent_stats,
        "total_runs": len(results),
        "failed_runs": total_expected - len(results),
    }
```

### 3.7 Output Generators

#### Output 1: Tabella probabilistica (con CI)

Esempio di output generato:

```
| Profilo Giudice              | Aggressivo              | Prudente                | Tecnico                 |
|------------------------------|-------------------------|-------------------------|-------------------------|
| Formalista + Pro Cass.       | A:10% [2-40%] R:30% X:60% | A:5% [0-30%] R:40% X:55% | A:15% [3-45%] R:35% X:50% |
| Formalista + Anti Cass.      | A:40% [15-70%] R:45% X:15% | A:25% [8-55%] R:55% X:20% | A:45% [18-73%] R:40% X:15% |
| Sostanzialista + Pro Cass.   | A:5% [0-30%] R:20% X:75%  | A:5% [0-30%] R:25% X:70%  | A:10% [2-40%] R:20% X:70%  |
| Sostanzialista + Anti Cass.  | A:35% [12-65%] R:50% X:15% | A:20% [5-50%] R:60% X:20% | A:30% [10-60%] R:50% X:20% |

A = annullamento, R = riqualificazione, X = rigetto, [CI 95%] = Wilson score interval
N=5 per cella — intervalli ampi, interpretare con cautela
```

#### Output 2: Decision tree

```python
def generate_decision_tree(aggregated: dict) -> dict:
    tree = {
        "question": "Profilo probabile del giudice?",
        "branches": []
    }

    for judge_profile in aggregated["style_effectiveness"]:
        # Stile con miglior success rate per questo profilo
        best_style = max(
            aggregated["style_effectiveness"],
            key=lambda s: aggregated["style_effectiveness"][s]
                         ["by_judge_profile"].get(judge_profile, {})
                         .get("success_rate", 0)
        )
        # ...costruzione branch con raccomandazione + CI

    # Dominated strategy detection
    for style_a in aggregated["style_effectiveness"]:
        for style_b in aggregated["style_effectiveness"]:
            if style_a == style_b:
                continue
            a_rates = aggregated["style_effectiveness"][style_a]["by_judge_profile"]
            b_rates = aggregated["style_effectiveness"][style_b]["by_judge_profile"]
            if all(a_rates.get(jp, {}).get("success_rate", 0) <=
                   b_rates.get(jp, {}).get("success_rate", 0)
                   for jp in a_rates):
                # style_a è dominato da style_b → segnalare

    return tree
```

#### Output 3: Memo strategico (Synthesizer agent)

Il memo è generato da un LLM separato (Synthesizer) che riceve i dati aggregati. Prompt:

```
Sei il consulente strategico di ATHENA. Hai i risultati di {n_runs} simulazioni
con variazione parametrica di profili giudice e stili di advocacy.

Produci un memo strategico per l'avvocato:

1. SINTESI ESECUTIVA (3-5 frasi) — esito più probabile, range probabilità, raccomandazione
2. ANALISI PER SCENARIO — per profilo giudice: probabilità, strategia ottimale, argomenti chiave
3. ARGOMENTI: RANKING DI EFFICACIA — universali vs polarizzanti vs irrilevanti
4. ANALISI DEL PRECEDENTE — tasso adesione vs distinguishing per profilo
5. RACCOMANDAZIONE STRATEGICA — dominante o condizionale, expected value in EUR
6. RISCHI E CAVEAT — limiti simulazione, fattori non modellati, gaps

Vincoli: scrivi per un avvocato, usa i numeri ma spiega cosa significano,
non mascherare incertezza. 1500-2500 parole.

NOTA: I confidence intervals sono ampi (N=5 per cella). Segnala esplicitamente
dove i dati sono insufficienti per una raccomandazione forte vs dove il segnale
è chiaro nonostante il campione ridotto.
```

### 3.8 Architettura complessiva

```
                    ┌──────────────────────┐
                    │    case.yaml         │
                    │  (fascicolo caso)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    │  simulation.yaml     │
                    │  (config parametri)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Monte Carlo         │
                    │  Orchestrator        │
                    │                      │
                    │  for each combo:     │
                    │  ┌────────────────┐  │
                    │  │  LangGraph     │  │
                    │  │  Single Run    │  │
                    │  │                │  │
                    │  │  ctx_app ──▶   │  │
                    │  │  appellant ──▶ │  │
                    │  │  validate ──▶  │  │
                    │  │  ctx_resp ──▶  │  │
                    │  │  respondent ──▶│  │
                    │  │  validate ──▶  │  │
                    │  │  ctx_judge ──▶ │  │
                    │  │  judge ──▶     │  │
                    │  │  validate      │  │
                    │  └───────┬────────┘  │
                    │          │            │
                    │    results[]          │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Aggregator        │
                    │  (con Wilson CI)     │
                    │                      │
                    │  probability_table   │
                    │  argument_scores     │
                    │  style_effectiveness │
                    │  precedent_stats     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │ Tabella        │ │ Decision     │ │ Memo         │
    │ Probabilistica │ │ Tree         │ │ Strategico   │
    │ (con CI)       │ │ (JSON+viz)   │ │ (LLM-gen)    │
    └────────────────┘ └──────────────┘ └──────────────┘

    ─────────────── Langfuse traces ──────────────────
```

### 3.9 Prerequisito: Capability Test del modello (fix C3)

Prima di implementare il grafo, validare il modello LLM locale:

```
Step 0: Capability Test
1. Prendere il prompt del giudice con contesto di esempio hardcoded
2. Testare con 2-3 modelli MLX candidati (Llama 3 70B, Qwen 2.5 72B, Mistral Large)
3. Metriche da verificare:
   - Tasso JSON valido (target: >90% senza retry)
   - Differenziazione tra profili (il giudice pro-Cassazione decide diversamente?)
   - Qualità reasoning giuridico (validazione umana dall'avvocato)
   - Rispetto referential integrity (usa solo ID forniti?)
4. Se nessun modello locale supera le soglie: fallback su API cloud
5. Documentare il modello scelto e i risultati del test
```

### Scelte architetturali — simulazione

1. **Context builders come layer separato** — l'information asymmetry è nel codice, non nel prompt. Verificabile e testabile indipendentemente.
2. **Sanitization filed_brief / internal_analysis** — previene information leakage. Il convenuto e il giudice vedono solo ciò che è "depositato", non l'analisi strategica interna. (fix C1)
3. **Validation con retry** — errori JSON recuperabili con feedback. Max 2 retry. Run fallite escluse ma contate.
4. **Warnings vs errors** — errori bloccano, warnings segnalano. Warnings aggregati nel memo.
5. **Monte Carlo loop esterno al grafo** — il grafo è una singola run. Parallelismo ortogonale.
6. **Synthesizer come agente separato** — il memo è un LLM che analizza, non un agente del trial.
7. **Langfuse per osservabilità** — ogni run = trace, ogni nodo = span.
8. **Dominated strategy detection** — se uno stile è sempre peggiore di un altro, viene identificato.
9. **Expected value calcolabile** — con probability_table + stakes: `EV = P(ann) × risparmio_totale + P(ricl) × risparmio_parziale - P(rig) × costo_causa`.
10. **Wilson confidence intervals** — con N=5, i CI sono ampi ma informativi. Mostrati nella tabella e interpretati dal Synthesizer. (fix C2)
11. **Capability test come step 0** — validazione del modello LLM prima di implementare. (fix C3)
12. **Temperatura fissa per ruolo** — giudice più bassa (0.3) per consistenza, parti più alta (0.4-0.5) per variabilità argomentativa. (fix I3)

---

## Limitazioni note

| ID | Limitazione | Impatto | Mitigazione |
|---|---|---|---|
| L1 | Convenuto senza stile parametrico | Testa solo "quale stile dell'opponente funziona" contro un convenuto standard | Documentare nel memo. Futuro: aggiungere respondent_profiles |
| L2 | Nessuna baseline di calibrazione | Non verificabile se i risultati sono accurati | Sanity check interno (coerenza profili) + validazione avvocato |
| L3 | Giudice non solleva d'ufficio | Potrebbe sottostimare probabilità accoglimento | Documentare nel memo |
| L4 | EV assume risk neutrality | Non modella avversione al rischio del cliente | Documentare. Futuro: utility function |
| L5 | No sensitivity su seed_arguments | Non testa "cosa succede senza argomento X" | Futuro: ablation test |
| L6 | N=5 per cella statisticamente limitato | CI ampi, raccomandazioni indicative | Wilson CI mostrati, Synthesizer interpreta cautamente |
