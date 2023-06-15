from dataclasses import dataclass, asdict
from typing import (
    Optional,
    Iterable, 
    Union, 
    Counter, 
    List, 
    TypedDict, 
    overload, 
    Tuple
    )
from enum import IntEnum
import datetime
import math
import weakref
import collections
import abc


@dataclass(frozen=True)
# @dataclass를 사용할 때 __repr__()는자동 생성한다.
class Sample:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


#x = sample(1, 2, 3, 4)

@dataclass(frozen=True)
class KnownSample(Sample):
    spicies: str


@dataclass(frozen=True)
class UnknownSample(Sample):
    pass


@dataclass
class TestingKnownSample:
    sample: KnownSample
    classification: Optional[str] = None


@dataclass(frozen=True)
class TrainingKnownSample:
    #classification 변수를 사용할 수 없음.
    sample: KnownSample


@dataclass
class ClassifiedSample:
    sample: Sample
    classfication: str


class Distance:
    def distance(self, s1: float, s2:float) -> float:
        raise NotImplementedError
    

class CD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
         return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )
    

class ED(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return math.hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )
    

class MD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum([
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width)
        ]) 
    

class SD(Distance):
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s1.petal_length),
                abs(s1.petal_width - s2.petal_width)
            ] / sum(
               [
                 s1.sepal_length + s2.sepal_length,
                 s1.sepal_width + s2.sepal_width,
                 s1.petal_length + s2.petal_length,
                 s1.petal_width + s2.petal_width    
               ]
            )
        )
    

@dataclass
class Hyperparameter:

    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]


    def classify(self, sample: Union[UnknownSample, TestingKnownSample]) -> str:
        """TODO: K-NN 알고리즘 구현"""
        training_data = self.data
        if not training_data:
            raise RuntimeError("No TrainingData")
        distances: list[tuple[float, TrainingKnownSample]] = sorted(
                (self.algorithm.distance(sample, known), known)
                for known in training_data.training
        )
        k_nearest: tuple[str] = (known.species for _, known in distances[:self.k])
        frequency : Counter[str] = collections.Counter(k_nearest)
        best_fit, *other = frequency.most_common()
        species, votes = best_fit
        return species


class SampleDict(TypedDict):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


class SamplePartition(List[SampleDict], abc.ABC):
    @overload
    def __init__(self, *, training_subset: float = 0.80) -> None:
        ...

    @overload
    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
    ) -> None:
        ...
    

    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]] = None,
            *,
            training_subset: float = 0.80
    ) -> None:
        self.training_subset = training_subset
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__() 


"""chapter 6에서 제시된, 분할을 위한 SamplePartion의 두 가지 전략 중 DealingPartition 구현"""
class DealingPartition(abc.ABC):
    @abc.abstractmethod
    def __init__(
            self,
            iterable: Optional[Iterable[SampleDict]],
            *,
            training_subset: float = 0.80
    ) -> None:
        ...

    
    @abc.abstractmethod
    def extend(self, items: Iterable[SampleDict]) -> None:
        ...

    
    @abc.abstractmethod
    def append(self, items: SampleDict) -> None:
        ...

    @property
    @abc.abstractmethod
    def training(self)-> List[TrainingKnownSample]:
        ...

    @property
    @abc.abstractmethod
    def tesing(self) -> List[TestingKnownSample]:
        ...


class CountingDealingPartition(DealingPartition):
    def __init__(
            self, 
            items: Optional[Iterable[SampleDict]],  
            *, 
            training_subset: Tuple[int, int] = (8, 10)
        ) -> None:
          self.training_subset = training_subset
          self.counter = 0
          self._training: List[TrainingKnownSample] = []
          self._testing: List[TestingKnownSample] = []
          if items:
              self.extend(items)
    

    def extend(self, items: Iterable[SampleDict]) -> None:
        for item in items:
            self.append(item)

    
    def append(self, item: SampleDict) -> None:
        n, d = self.training_subset
        if self.counter % d < n:
            self._training.append(TrainingKnownSample(**item))
        else:
            self._testing.append(TestingKnownSample(**item))
        self.counter += 1

    
    @property
    def training(self) -> List[TrainingKnownSample]:
        return self._training
    

    @property
    def tesing(self) -> List[TestingKnownSample]:
        return self._tesing
        
        


class TrainingData:
    def __init__(self, name) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[KnownSample] = []
        self.testing: list[KnownSample] = []
        self.tuning: list[Hyperparameter] = []
 

    def load(self, raw_data_source : Iterable[dict[str,str]]) -> None: 
        for n, row in enumerate(raw_data_source):
            if n % 5 == 0:
                test = TestingKnownSample(
                     species = row["species"],
                     sepal_length = float(row["sepal_length"]),
                     sepal_width = float(row["sepal_width"]),
                     petal_length = float(row["petal_length"]),
                     petal_width = float(row["petal_width"])
                     
                )
                self.testing.append(test)
            else:
                train = TrainingKnownSample(
                     species = row["species"],
                     sepal_length = float(row["sepal_length"]),
                     sepal_width = float(row["sepal_width"]),
                     petal_length = float(row["petal_length"]),
                     petal_width = float(row["petal_width"])
                )
                self.testing.append(test)
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter : Hyperparameter) -> None: 
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter : Hyperparameter, sample : Sample) -> ClassifiedSample:  
        return ClassifiedSample(
            classification = parameter.classify(sample), sample = sample
        )
