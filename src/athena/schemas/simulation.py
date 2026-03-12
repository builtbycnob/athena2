import math
from functools import reduce
from operator import mul

from pydantic import BaseModel


class PartyProfile(BaseModel):
    id: str
    party_id: str           # which party this profile applies to
    role_type: str           # "advocate" | "adjudicator"
    parameters: dict         # style, jurisprudential_orientation, formalism, etc.


class SimulationConfig(BaseModel):
    case_ref: str
    language: str
    judge_profiles: list[PartyProfile]
    party_profiles: dict[str, list[PartyProfile]]  # party_id → [profiles]
    temperatures: dict[str, float]                   # party_id → temp
    runs_per_combination: int

    @property
    def total_runs(self) -> int:
        party_counts = [len(profiles) for profiles in self.party_profiles.values()]
        party_product = reduce(mul, party_counts, 1)
        return len(self.judge_profiles) * party_product * self.runs_per_combination


def migrate_simulation_v1(raw: dict) -> dict:
    """Convert old-format simulation config to N-party format.

    Old format: appellant_profiles, temperature (with appellant/respondent/judge keys).
    New format: party_profiles dict, temperatures dict.
    """
    # Already new format
    if "party_profiles" in raw:
        return raw

    result = dict(raw)

    # Migrate appellant_profiles → party_profiles
    if "appellant_profiles" in result:
        old_profiles = result.pop("appellant_profiles")
        # Convert old AppellantProfile format to PartyProfile format
        party_profiles = {}
        party_profiles["opponente"] = [
            {
                "id": p["id"],
                "party_id": "opponente",
                "role_type": "advocate",
                "parameters": {"style": p["style"]},
            }
            for p in old_profiles
        ]
        result["party_profiles"] = party_profiles

    # Migrate judge_profiles to PartyProfile format
    if "judge_profiles" in result:
        old_judges = result["judge_profiles"]
        result["judge_profiles"] = [
            {
                "id": j["id"],
                "party_id": "judge",
                "role_type": "adjudicator",
                "parameters": {
                    k: v for k, v in j.items() if k != "id"
                },
            }
            for j in old_judges
        ]

    # Migrate temperature → temperatures
    if "temperature" in result and "temperatures" not in result:
        old_temp = result.pop("temperature")
        result["temperatures"] = old_temp

    return result
