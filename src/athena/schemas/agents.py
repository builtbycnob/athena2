from pydantic import BaseModel, field_validator


# --- Appellant ---

class PrecedentAddress(BaseModel):
    id: str
    strategy: str  # distinguish | criticize | limit_scope
    reasoning: str


class Argument(BaseModel):
    id: str
    type: str  # derived | new
    derived_from: str | None = None
    claim: str
    legal_reasoning: str
    norm_text_cited: list[str]
    facts_referenced: list[str]
    evidence_cited: list[str]
    precedents_addressed: list[PrecedentAddress] = []
    supports: str | None = None


class AppellantRequests(BaseModel):
    primary: str
    subordinate: str


class AppellantFiledBrief(BaseModel):
    arguments: list[Argument]
    requests: AppellantRequests

    @field_validator("arguments")
    @classmethod
    def at_least_one_argument(cls, v):
        if len(v) == 0:
            raise ValueError("Brief must contain at least one argument")
        return v


class AppellantInternalAnalysis(BaseModel):
    strength_self_assessments: dict[str, float]
    key_vulnerabilities: list[str]
    strongest_point: str
    gaps: list[str]


class AppellantBrief(BaseModel):
    filed_brief: AppellantFiledBrief
    internal_analysis: AppellantInternalAnalysis


# --- Respondent ---

class PrecedentCitation(BaseModel):
    id: str
    relevance: str


class PreliminaryObjection(BaseModel):
    id: str
    type: str  # tardività | inammissibilità | incompetenza | difetto_legittimazione
    claim: str
    legal_basis: list[str]
    reasoning: str


class ResponseToOpponent(BaseModel):
    to_argument: str
    counter_strategy: str  # rebut | distinguish | concede_partially
    counter_reasoning: str
    norm_text_cited: list[str]
    precedents_cited: list[PrecedentCitation] = []


class AffirmativeDefense(BaseModel):
    id: str
    type: str  # derived | new
    derived_from: str | None = None
    claim: str
    legal_reasoning: str
    norm_text_cited: list[str]
    facts_referenced: list[str]
    evidence_cited: list[str]


class RespondentRequests(BaseModel):
    primary: str
    fallback: str


class RespondentFiledBrief(BaseModel):
    preliminary_objections: list[PreliminaryObjection]
    responses_to_opponent: list[ResponseToOpponent]
    affirmative_defenses: list[AffirmativeDefense]
    requests: RespondentRequests


class RespondentInternalAnalysis(BaseModel):
    strength_self_assessments: dict[str, float]
    key_vulnerabilities: list[str]
    opponent_strongest_point: str
    gaps: list[str]


class RespondentBrief(BaseModel):
    filed_brief: RespondentFiledBrief
    internal_analysis: RespondentInternalAnalysis


# --- Judge ---

class PreliminaryObjectionRuling(BaseModel):
    objection_id: str
    sustained: bool
    reasoning: str


class ArgumentEvaluation(BaseModel):
    argument_id: str
    party: str  # appellant | respondent
    persuasiveness: float
    strengths: str
    weaknesses: str
    determinative: bool


class PrecedentAnalysisItem(BaseModel):
    followed: bool
    distinguished: bool
    reasoning: str


class IncorrectQualification(BaseModel):
    consequence: str  # annulment | reclassification
    consequence_reasoning: str
    applied_norm: str
    sanction_determined: int
    points_deducted: int


class Verdict(BaseModel):
    qualification_correct: bool
    qualification_reasoning: str
    if_incorrect: IncorrectQualification | None = None
    costs_ruling: str


class JudgeDecision(BaseModel):
    preliminary_objections_ruling: list[PreliminaryObjectionRuling]
    case_reaches_merits: bool
    argument_evaluation: list[ArgumentEvaluation]
    precedent_analysis: dict[str, PrecedentAnalysisItem]
    verdict: Verdict
    reasoning: str
    gaps: list[str]
