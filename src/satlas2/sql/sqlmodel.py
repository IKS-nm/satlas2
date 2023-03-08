from typing import Optional

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ['Result', 'Metadata', 'Base']


class Base(DeclarativeBase):
    pass


class Result(Base):
    __tablename__ = "result"

    Source: Mapped[str] = mapped_column(primary_key=True)
    Model: Mapped[str] = mapped_column(primary_key=True)
    Parameter: Mapped[str] = mapped_column(primary_key=True)
    Value: Mapped[float]
    Stderr: Mapped[float]
    Minimum: Mapped[float]
    Maximum: Mapped[float]
    Expression: Mapped[Optional[str]]
    Vary: Mapped[bool]

    def __repr__(self):
        return f"Result(Source={self.Source!r}, Model={self.Model!r}, Parameter={self.Parameter!r})"


class Metadata(Base):
    __tablename__ = "metadata"

    Source: Mapped[str] = mapped_column(primary_key=True)
    Fitting_method: Mapped[str] = mapped_column(primary_key=True)
    Message: Mapped[str]
    Function_evaluations: Mapped[int]
    Data_points: Mapped[int]
    Variables: Mapped[int] = mapped_column(primary_key=True)
    Chisquare: Mapped[float]
    Redchi: Mapped[float]
    Aic: Mapped[float]
    Bic: Mapped[float]

    def __repr__(self):
        return f"Metadata(Source={self.Source!r}, Fitting method={self.Fitting_method!r}, Variables={self.Variables!r})"
