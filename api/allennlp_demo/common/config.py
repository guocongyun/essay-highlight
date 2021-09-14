import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from allennlp.predictors import Predictor


VALID_ATTACKERS = ["hotflip", "input_reduction"]
VALID_INTERPRETERS = ["simple_gradient", "smooth_gradient", "integrated_gradient"]


@dataclass(frozen=True)
class Model:
    """
    Class capturing the options we support per model.
    """

    id: str
    """
    A unique name to identify each demo.
    """

    archive_file: str
    """
    The path to the model's archive_file.
    """

    pretrained_model_id: Optional[str] = None
    """
    The ID of a pretrained model to use from `allennlp_models.pretrained`.
    """

    predictor_name: Optional[str] = None
    """
    Optional predictor name to override the default predictor associated with the archive.

    This is ignored if `pretrained_model_id` is given.
    """

    overrides: Optional[Dict[str, Any]] = None
    """
    Optional parameter overrides to pass through when loading the archive.

    This is ignored if `pretrained_model_id` is given.
    """

    attackers: List[str] = field(default_factory=lambda: VALID_ATTACKERS)
    """
    List of valid attackers to use.
    """

    interpreters: List[str] = field(default_factory=lambda: VALID_INTERPRETERS)
    """
    List of valid interpreters to use.
    """

    use_old_load_method: bool = False
    """
    Some models that run on older versions need to be load differently.
    """

    qa_model_path: str = ""
    """
    Location of question answering model's pytorch bin file
    """

    summarize_model_path: str = ""
    """
    Location of summarization model's pytorch bin file
    """

    similarity_model_path: str = ""
    """
    Location of sentence transformer model's pytorch bin file
    """

    max_seq_length: int = 512
    """
    Max sequence length of input ids
    """

    stride: int = 512
    """
    The stride used when handling longer than 512 inputs
    """

    pad_on_right: bool = True
    """
    Wether to pad on right or not
    """

    start_weight: float = 0.5
    """
    0 to 1 value for the importance of the start token prediction, will ignore end token if set to 1
    """

    nbest: int = 10
    """
    Number of best start and end token prediction to consider when determining the answer, lowering this reduces prediction time and accuracy
    """

    max_answer_length: int = 300
    """
    Maximum amount of span to consider when giving prediction
    """

    min_answer_length: int = 5,
    """
    Minimum amount of span to consider when giving prediction
    """

    similarity_model_weight: int = 0,
    """
    Ratio of similarity model and question answering model's weight in prediction, setting it to 1 ignores question answering and setting it to 0 ignores text similarity
    """

    summerization_model: bool = False,
    """
    True if model is used for summerization, else false.
    """

    @classmethod
    def from_file(cls, path: str) -> "Model":
        with open(path, "r") as fh:
            raw = json.load(fh)
            if "pretrained_model_id" in raw:
                from allennlp_models.pretrained import get_pretrained_models

                model_card = get_pretrained_models()[raw["pretrained_model_id"]]
                raw["archive_file"] = model_card.model_usage.archive_file
                raw["predictor_name"] = model_card.registered_predictor_name
            out = cls(**raw)


        return out

    def load_predictor(self) -> Predictor:
        if self.pretrained_model_id is not None:
            from allennlp_models.pretrained import load_predictor

            return load_predictor(self.pretrained_model_id, overrides=self.overrides)

        assert self.archive_file is not None

        if self.use_old_load_method:
            from allennlp.models.archival import load_archive

            # Older versions require overrides to be passed as a JSON string.
            o = json.dumps(self.overrides) if self.overrides is not None else None
            archive = load_archive(self.archive_file, overrides=o)
            return Predictor.from_archive(archive, self.predictor_name)

        return Predictor.from_path(
            self.archive_file, predictor_name=self.predictor_name, overrides=self.overrides
        )
