# ATHENA — Analisi Strategica Simulata
## Caso: GDP Milano 17928/2025 — Violazione CdS

**Data analisi:** 11 Marzo 2026
**Simulazioni completate:** 55/60 (92%)
**Metodo:** Simulazione Monte Carlo con agenti LLM (appellant, respondent, judge) su 4 profili giudice x 3 stili argomentativi x 5 ripetizioni

> **Nota:** Questo report è generato da un sistema sperimentale di simulazione strategica.
> NON costituisce parere legale. I risultati vanno interpretati come indicazioni di tendenza,
> non come previsioni certe. Gli intervalli di confidenza sono ampi (N=4-5 per cella).

---

## 1. Matrice di Probabilità degli Esiti

| Profilo Giudice | Aggressivo | Prudente | Tecnico |
|-----------------|------------|----------|---------|
| **Formalista anti-Cass.** | A:20% R:60% X:20% | A:0% R:80% X:20% | A:0% R:60% X:40% |
| **Formalista pro-Cass.** | A:0% R:0% X:100% | A:0% R:20% X:80% | A:0% R:0% X:100% |
| **Sostanzialista anti-Cass.** | A:0% R:25% X:75% | A:0% R:50% X:50% | A:0% R:20% X:80% |
| **Sostanzialista pro-Cass.** | A:0% R:0% X:100% | A:0% R:40% X:60% | A:0% R:0% X:100% |

**Legenda:** A = annullamento, R = riqualificazione, X = rigetto
**Intervalli di confidenza (Wilson 95%):** ampi per via del campione ridotto — vedere dettaglio nel decision tree

---

## 2. Albero Decisionale

```
Profilo probabile del giudice?
│
├─ Formalista anti-Cassazione
│  ├─ Aggressivo:  successo 80% [38%-96%]  ★ RACCOMANDATO
│  │    A:20% R:60% X:20%
│  ├─ Prudente:    successo 80% [38%-96%]
│  │    A:0% R:80% X:20%
│  └─ Tecnico:     successo 60% [23%-88%]
│       A:0% R:60% X:40%
│
├─ Formalista pro-Cassazione
│  ├─ Aggressivo:  successo 0% [0%-49%]
│  ├─ Prudente:    successo 20% [4%-62%]  ★ MENO PEGGIO
│  └─ Tecnico:     successo 0% [0%-49%]
│
├─ Sostanzialista anti-Cassazione
│  ├─ Aggressivo:  successo 25% [5%-70%]
│  ├─ Prudente:    successo 50% [15%-85%]  ★ RACCOMANDATO
│  └─ Tecnico:     successo 20% [4%-62%]
│
└─ Sostanzialista pro-Cassazione
   ├─ Aggressivo:  successo 0% [0%-49%]
   ├─ Prudente:    successo 40% [12%-77%]  ★ MENO PEGGIO
   └─ Tecnico:     successo 0% [0%-43%]
```

---

## 3. Ranking Efficacia Argomenti

| # | Argomento | Persuasività media | Determinativo | Note |
|---|-----------|-------------------|---------------|------|
| 1 | **ARG1** | 0.81 (±0.19) | 85% | Efficace trasversalmente su tutti i profili |
| 2 | RARG1 | 0.75 (±0.26) | 80% | Forte con pro-Cass. (0.87), debole con anti-Cass. (0.52) |
| 3 | ARG2 | 0.68 (±0.22) | 18% | Persuasivo ma raramente determinativo |
| 4 | ARG3 | 0.59 (±0.24) | 21% | Secondario |
| 5 | ARG4 | 0.45 (±0.24) | 0% | Irrilevante — campione insufficiente |
| 6 | RARG2 | 0.45 (±0.07) | 0% | Irrilevante — campione insufficiente |

---

## 4. Analisi del Precedente (Cass. 16515/2005)

| Profilo Giudice | Adesione al precedente | Distinguishing |
|-----------------|----------------------|----------------|
| Formalista pro-Cass. | **92%** | 8% |
| Sostanzialista pro-Cass. | **86%** | 14% |
| Sostanzialista anti-Cass. | 69% | 31% |
| Formalista anti-Cass. | 27% | **73%** |

**Media globale:** 67% adesione, 33% distinguishing.

---

## 5. Sintesi Strategica

### Osservazioni principali

1. **Il rigetto è l'esito dominante** (~65-80% aggregato). Il caso è strutturalmente in salita.
2. **Il profilo del giudice è il fattore determinante**: la differenza tra anti-Cassazione e pro-Cassazione vale un delta di 60-80 punti percentuali sull'esito.
3. **ARG1 è l'unico argomento con impatto consistente** su tutti i profili (0.81 persuasività, 85% determinativo).
4. **Il precedente Cass. 16515/2005** è quasi vincolante per giudici pro-Cass. (92% adesione) ma distinguibile con giudici anti-Cass. (73% distinguishing).

### Raccomandazioni dal sistema

- **Stile aggressivo/tecnico** centrato su ARG1 come pilastro della difesa
- **RARG1** da usare solo se il giudice è pro-Cassazione (altrimenti inefficace)
- **ARG2-4**: supporto secondario, non determinativi
- **Strategia "prudente" risulta dominata** nelle simulazioni (ma questo potrebbe essere un artefatto — vedere caveat)
- **Preparare due versioni della strategia**: una aggressiva (se giudice anti-Cass.) e una di contenimento costi (se pro-Cass.)

### Caveat importanti

- Campione ridotto (N=4-5 per cella) → intervalli di confidenza ampi
- Non modella: qualità prove documentali, fattore umano in udienza, cambi giurisprudenziali
- Il labeling "prudente dominata" potrebbe non riflettere la realtà processuale
- Risultati da interpretare come tendenze, non previsioni

---

## 6. Parametri della Simulazione

| Parametro | Valore |
|-----------|--------|
| Simulazioni completate | 55/60 (5 fallite per errore parsing) |
| Chiamate LLM totali | 238 |
| Token generati | 494.000 |
| Tempo totale | 103 minuti |
| Velocità media | 79.8 tok/s |
| Modello | Qwen3.5-35B-A3B (locale, MLX) |
| Profili giudice | formalista pro/anti-Cass., sostanzialista pro/anti-Cass. |
| Stili appellant | aggressivo, prudente, tecnico |
| Ripetizioni per cella | 5 |

---

*Report generato da ATHENA v0.1.1 — Sistema sperimentale di simulazione strategica legale*
*NON costituisce parere legale*
