from pydantic import BaseModel


class JudgeProfile(BaseModel):
    id: str
    jurisprudential_orientation: str  # follows_cassazione | distinguishes_cassazione
    formalism: str  # high | low


class AppellantProfile(BaseModel):
    id: str
    style: str


class TemperatureConfig(BaseModel):
    appellant: float
    respondent: float
    judge: float


class SimulationConfig(BaseModel):
    case_ref: str
    language: str
    judge_profiles: list[JudgeProfile]
    appellant_profiles: list[AppellantProfile]
    temperature: TemperatureConfig
    runs_per_combination: int

    @property
    def total_runs(self) -> int:
        return len(self.judge_profiles) * len(self.appellant_profiles) * self.runs_per_combination
