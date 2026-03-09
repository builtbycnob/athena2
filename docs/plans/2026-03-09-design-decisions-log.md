# ATHENA PoC — Design Decisions Log

Registro delle decisioni architetturali prese durante il design, con razionale e alternative scartate.

## Decisione 1: Approccio Monte Carlo vs incrementale

**Scelta**: Progettare per Monte Carlo (approccio C) da subito, validare con N=1.

**Alternative considerate**:
- (A) Debate strutturato — singolo round, semplice ma non produce distribuzione probabilistica
- (B) Multi-round iterativo — più realistico ma overengineering per GdP

**Razionale**: L'utente ha correttamente osservato che progettare per A e poi estendere a C rischia vincoli architetturali. Analisi: il costo di retrofit da A a C è basso (C contiene A), ma progettare lo schema e i prompt pensando a C da subito evita rework su output schema, parametri simulazione e formato del giudice. Compromesso: architettura C, primo test con N=1.

## Decisione 2: Case file e simulation config separati

**Scelta**: Due file YAML distinti — case.yaml (stabile) e simulation.yaml (varia).

**Alternativa scartata**: Tutto in un unico file.

**Razionale**: Vuoi poter lanciare simulazioni diverse sullo stesso caso. La separazione rende lo schema più pulito e il caso riusabile.

## Decisione 3: Agenti generativi ibridi (seed_arguments)

**Scelta**: Il case file contiene seed_arguments (spunti sintetici). Gli agenti sono liberi di espandere, riorganizzare, aggiungere nuovi argomenti, scartare quelli deboli.

**Alternative considerate**:
- Agenti puramente generativi — ricevono solo fatti + norme, generano tutto. Rischio: meno controllabile.
- Agenti puramente presentativi — riformattano argomenti pre-scritti. Rischio: non scoprono nulla di nuovo.

**Razionale**: L'ibrido dà controllo (i seed guidano) senza togliere capacità generativa. Il campo "type": "derived | new" rende tracciabile cosa viene dal seed e cosa è generato.

## Decisione 4: Giudice = solo decisore, no probabilità

**Scelta**: Il giudice produce solo verdict + reasoning. Le probabilità emergono dall'aggregazione delle N run Monte Carlo.

**Alternativa scartata**: Il giudice produce sia verdict che outcome_probabilities.

**Razionale**: Un giudice decide, non stima probabilità. Chiedere allo stesso agente di essere decisore e analista crea un conflitto di ruolo. Le probabilità sono una proprietà emergente della simulazione, non un output del singolo agente. L'aggregatore le calcola contando i verdict su N run.

## Decisione 5: Verdict a due livelli

**Scelta**: Il giudice risponde a due questioni distinte:
1. `qualification_correct` — la qualificazione nel verbale è corretta?
2. `if_incorrect.consequence` — se errata, annullamento o riqualificazione?

**Alternativa scartata**: Unico campo `outcome: accepted | rejected | partial`.

**Razionale**: Sono questioni giuridiche distinte con standard diversi. Fonderle nasconde il ragionamento del giudice e rende l'aggregazione meno informativa. Sapere che il 70% dei giudici ritiene la qualificazione errata ma solo il 40% annulla (il resto riqualifica) è un'informazione strategica cruciale.

## Decisione 6: Stile advocacy parametrico

**Scelta**: 3 profili opponente (aggressivo, prudente, tecnico) come parametro della simulazione, cross-product con 4 profili giudice.

**Alternativa scartata**: Stile fisso dell'opponente.

**Razionale**: ATHENA deve dire all'avvocato quale stile funziona meglio con quale tipo di giudice. Senza parametrizzare lo stile, la domanda strategica resta senza risposta. La matrice 4×3 = 12 combinazioni × 3 run = 36 run totali è gestibile su MLX locale.

## Decisione 7: Grounding normativo obbligatorio

**Scelta**: Sezione `legal_texts` nel case file con testi completi delle norme. Prompt: "Ragiona ESCLUSIVAMENTE sui testi forniti."

**Alternativa scartata**: Affidarsi alla conoscenza pre-training del modello.

**Razionale**: Il pre-training può contenere versioni obsolete delle norme o inventare articoli. In ambito legale, l'accuratezza del testo è critica. Il costo (più token in input) è accettabile per la correttezza.

## Decisione 8: Referential integrity + campo gaps

**Scelta**: Gli agenti possono referenziare SOLO ID presenti nel fascicolo. Se manca qualcosa, lo segnalano in "gaps".

**Razionale**: Impedisce allucinazioni di ID inesistenti. Il campo "gaps" è un bonus: segnala cosa manca nel case file, guidando l'utente a completarlo.

## Decisione 9: Profili giudice come matrice 2×2

**Scelta**: Due assi ortogonali:
- Orientamento giurisprudenziale: follows_cassazione vs distinguishes_cassazione
- Formalismo: high vs low

**Razionale**: Copre lo spazio dei comportamenti giudiziali rilevanti per questo caso senza esplosione combinatoria. L'utente ha scelto esplicitamente entrambi gli assi.

## Decisione 10: Lingua italiana per prompt e output

**Scelta**: Prompt in italiano, output in italiano. Parametro `language` nella simulation config per futuri test comparativi.

**Razionale**: Il reasoning giuridico italiano ha formulazioni specifiche. Un mismatch linguistico degrada la qualità. Il parametro `language` permette di testare se il modello ragiona meglio in inglese, senza hardcodare la scelta.

## Decisione 11: Eccezioni preliminari prima del merito

**Scelta**: Il prompt del convenuto include eccezioni di rito (tardività, inammissibilità, etc.) come primo step. Il giudice le valuta prima di entrare nel merito.

**Alternativa scartata**: Solo merito, no eccezioni.

**Razionale**: Nella realtà processuale, un'eccezione preliminare fondata chiude il caso senza entrare nel merito. Ometterla sarebbe una simulazione incompleta. Il vincolo "se non fondate, lascia la lista vuota" impedisce eccezioni pretestuose.

## Decisione 12: Context builders come layer separato di information asymmetry

**Scelta**: Tre funzioni `build_context_*` che assemblano viste diverse del fascicolo per ogni agente. L'asimmetria informativa è nel codice Python, non nei prompt.

**Alternativa scartata**: Includere tutto nel prompt e dire "ignora ciò che non dovresti sapere".

**Razionale**: "Ignora X" non funziona con LLM — l'informazione presente nel contesto influenza il ragionamento anche se il prompt dice di ignorarla. La separazione a livello di codice è l'unico modo affidabile di garantire information asymmetry. Bonus: è testabile con unit test (verifica che il contesto dell'opponente non contenga beliefs del convenuto).

## Decisione 13: Validation con retry e feedback

**Scelta**: Tre livelli di validazione (schema JSON, referential integrity, completezza) con max 2 retry. Al retry, gli errori specifici vengono iniettati nel prompt. Al terzo fallimento, run marcata come error ed esclusa.

**Alternative considerate**:
- Nessuna validazione — rischio output inutilizzabili aggregati con quelli validi
- Validazione senza retry — troppo punitivo, un errore JSON è spesso recuperabile
- Retry infiniti — rischio loop, spreco di compute

**Razionale**: Il compromesso (2 retry) bilancia recuperabilità e costi. Il feedback nel prompt è più efficace del semplice "riprova": l'LLM vede esattamente cosa ha sbagliato. La distinzione errori/warnings evita di scartare run con problemi minori.

## Decisione 14: Monte Carlo loop esterno al grafo LangGraph

**Scelta**: Il grafo LangGraph modella una singola run (opponente → convenuto → giudice). Il loop che itera sulle combinazioni di parametri è Python esterno al grafo.

**Alternativa considerata**: Supergraph LangGraph con fan-out (Send API) per parallelismo interno.

**Razionale**: La singola run è il componente fondamentale. Tenerla come grafo autonomo semplifica testing (puoi testare una run senza Monte Carlo), debugging (ogni run ha il suo trace Langfuse), e manutenzione. Il parallelismo è ortogonale: basta sostituire il `for` con `asyncio.gather`, il grafo non cambia. La complessità di un supergraph non è giustificata per il PoC.

## Decisione 15: Synthesizer come agente LLM separato per il memo strategico

**Scelta**: Il memo strategico è generato da una chiamata LLM separata (Synthesizer) che riceve i dati aggregati. Non è un nodo del grafo di simulazione.

**Alternative considerate**:
- Template-based (no LLM) — deterministico ma incapace di interpretare pattern complessi
- Il giudice produce il memo — conflitto di ruolo, il giudice decide, non consiglia

**Razionale**: Il memo richiede capacità interpretative (quali pattern sono significativi? cosa dovrebbe fare l'avvocato?) che un template non può fornire. Separare il Synthesizer dal trial garantisce che l'analisi sia imparziale rispetto ai singoli agenti.

## Decisione 16: Giudice non vede advocacy_style

**Scelta**: Il context builder del giudice NON include `advocacy_style` come parametro esplicito. Il giudice percepisce lo stile solo attraverso la memoria dell'opponente.

**Razionale**: Un giudice reale non sa che strategia l'avvocato ha scelto consapevolmente, ne vede solo il risultato. Includere lo stile come metadato rischierebbe di influenzare il giudice-LLM in modo non realistico (es. "questo avvocato è aggressivo, quindi lo penalizzo").

## Decisione 17: Dominated strategy detection nell'aggregatore

**Scelta**: L'aggregatore verifica se uno stile di advocacy è strettamente dominato da un altro (mai migliore su nessun profilo giudice). Se sì, lo segnala esplicitamente.

**Razionale**: Identificare strategie dominate è il risultato più actionable della game theory: "non usare mai lo stile X" è una raccomandazione chiara e ad alto valore. Non richiede assunzioni sulla distribuzione dei profili giudice.

---

## Fix dalla review complessiva (2026-03-09)

### Fix C1: Separazione filed_brief / internal_analysis — information leakage

**Problema**: L'output degli agenti conteneva `strategy.key_vulnerabilities` e `gaps` visibili all'avversario e al giudice. In un procedimento reale, il convenuto vede solo la memoria depositata, non le note strategiche interne dell'avvocato.

**Fix applicato**:
- Output di ogni agente diviso in `filed_brief` (depositato, visibile a tutti) e `internal_analysis` (work product interno, visibile solo all'analisi strategica)
- `requests` (cosa si chiede al giudice) separato da `internal_analysis` — le richieste sono pubbliche
- Context builders filtrano: passano solo `filed_brief` all'avversario e al giudice
- Validation layer verifica la presenza di entrambi i blocchi
- `internal_analysis` usata solo dall'aggregatore e dal Synthesizer per il memo strategico

### Fix C2: N=5 con Wilson confidence intervals

**Problema**: N=3 per combinazione → una singola run anomala spostava la probabilità del 33%. Statisticamente insufficiente per raccomandazioni.

**Fix applicato**:
- `runs_per_combination` aumentato da 3 a 5 → 60 run totali (accettabile su MLX)
- Aggregatore calcola Wilson score intervals (robusti per piccoli campioni, a differenza degli intervalli normali)
- CI mostrati nella tabella probabilistica
- Synthesizer istruito a interpretare cautamente dove i CI sono ampi
- Tabella di esempio aggiornata con formato `P% [CI_low-CI_high%]`

### Fix C3: Capability test come step 0

**Problema**: Il design assume capacità del modello MLX (JSON strutturato, reasoning giuridico italiano, differenziazione profili) mai validate.

**Fix applicato**:
- Aggiunto step 0 esplicito nel design: test con 2-3 modelli candidati prima dell'implementazione
- Metriche definite: tasso JSON valido (>90%), differenziazione profili, qualità reasoning, referential integrity
- Fallback su API cloud se nessun modello locale supera le soglie

### Fix I1: Rimozione private_beliefs

**Problema**: `private_beliefs` nello schema ma mai iniettate nei contesti degli agenti. Dati morti.

**Fix applicato**: Rimosse dallo schema con commento. Se servono in futuro, saranno aggiunte con un design di come influenzano il comportamento degli agenti.

### Fix I2: Annotazione procedural_rules

**Problema**: `procedural_rules` nello schema ma non vincolanti nel grafo. Suggerivano estensibilità non presente.

**Fix applicato**: Commento aggiunto nello schema: "PoC: il grafo è hardcoded per questo rito. Futuro: le regole guideranno la costruzione dinamica del grafo."

### Fix I3: Temperatura fissa per ruolo

**Problema**: Temperatura random in [0.3, 0.7] aggiungeva rumore non interpretabile e non tracciato nell'aggregazione.

**Fix applicato**:
- `temperature_range` sostituito con `temperature` dict fisso per ruolo
- Giudice: 0.3 (più consistente nelle decisioni)
- Convenuto: 0.4
- Opponente: 0.5 (più variabilità argomentativa)
- Variabilità tra run garantita dal sampling naturale dell'LLM
