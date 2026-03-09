from pydantic import BaseModel, field_validator


class Precedent(BaseModel):
    id: str
    citation: str
    holding: str
    weight: str


class ProceduralRules(BaseModel):
    rite: str
    phases: list[str]
    allowed_moves: dict[str, list[str]]


class Jurisdiction(BaseModel):
    country: str
    court: str
    venue: str
    applicable_law: list[str]
    key_precedents: list[Precedent]
    procedural_rules: ProceduralRules


class PartyObjectives(BaseModel):
    primary: str
    subordinate: str


class Party(BaseModel):
    id: str
    role: str
    type: str
    objectives: PartyObjectives
    entity: str | None = None


class Sanction(BaseModel):
    norm: str
    fine_range: list[int]
    points_deducted: int


class Stakes(BaseModel):
    current_sanction: Sanction
    alternative_sanction: Sanction
    litigation_cost_estimate: int
    non_monetary: str | None = None


class Evidence(BaseModel):
    id: str
    type: str
    description: str
    produced_by: str
    admissibility: str
    supports_facts: list[str]


class Fact(BaseModel):
    id: str
    description: str
    evidence: list[str]


class DisputedFact(BaseModel):
    id: str
    description: str
    appellant_position: str
    respondent_position: str
    depends_on_facts: list[str]


class Facts(BaseModel):
    undisputed: list[Fact]
    disputed: list[DisputedFact]


class LegalText(BaseModel):
    id: str
    norm: str
    text: str


class SeedArgument(BaseModel):
    id: str
    claim: str
    direction: str
    references_facts: list[str]


class SeedArguments(BaseModel):
    appellant: list[SeedArgument]
    respondent: list[SeedArgument]


class TimelineEvent(BaseModel):
    date: str
    event: str


class CaseFile(BaseModel):
    case_id: str
    jurisdiction: Jurisdiction
    parties: list[Party]
    stakes: Stakes
    evidence: list[Evidence]
    facts: Facts
    legal_texts: list[LegalText]
    seed_arguments: SeedArguments
    key_precedents: list[Precedent]
    timeline: list[TimelineEvent]

    def extract_all_ids(self) -> set[str]:
        """Extract all valid IDs from the case file for referential integrity checks."""
        ids: set[str] = set()
        for e in self.evidence:
            ids.add(e.id)
        for f in self.facts.undisputed:
            ids.add(f.id)
        for f in self.facts.disputed:
            ids.add(f.id)
        for lt in self.legal_texts:
            ids.add(lt.id)
        for p in self.key_precedents:
            ids.add(p.id)
        for sa in self.seed_arguments.appellant:
            ids.add(sa.id)
        for sa in self.seed_arguments.respondent:
            ids.add(sa.id)
        return ids
