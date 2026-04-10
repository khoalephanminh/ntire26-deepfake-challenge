import logging

from transformers import PreTrainedModel
from svd_training.model_keys import get_mlp_names
from svd_training.svd_linear import SVDLinear

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SVDForCausalLM(PreTrainedModel):
    def merge(self):
        for layer_index in range(len(self.vision_model.encoder.layers)):
            for mlp_name in self._target_modules:
                try:
                    exec(
                        f"self.vision_model.encoder.layers[layer_index].{mlp_name} = self.vision_model.encoder.layers[layer_index].{mlp_name}.get_merged_linear()"
                    )
                except AttributeError:
                    continue

        self.visual_projection = self.visual_projection.get_merged_linear()

    @staticmethod
    def create_from_model(model, rank_fraction, target_modules=None):
        if target_modules is None:
            target_modules = get_mlp_names()
        model._target_modules = target_modules

        _logger.info(f"Building SVD model with rank_fraction={rank_fraction}")
        _logger.info(f"lm_head is substituted with rank_fraction={rank_fraction}")

        print("visual_projection=", model.visual_projection)
        model.visual_projection = SVDLinear.create_from_weight(
            model.visual_projection.weight, rank_fraction
        )
        print("visual_projection=", model.visual_projection)

        for layer_index in range(len(model.vision_model.encoder.layers)):
            for mlp_name in target_modules:
                try:
                    exec(f"weight = model.vision_model.encoder.layers[layer_index].{mlp_name}.weight")
                    exec(
                        f"model.vision_model.encoder.layers[layer_index].{mlp_name} = SVDLinear.create_from_weight(weight, rank_fraction)"
                    )
                except AttributeError:
                    continue

                _logger.info(
                    f"Layer {layer_index} on {mlp_name} with rank_fraction={rank_fraction} is substituted"
                )
        _logger.info("Adding merge method to the model.")
        model.merge = SVDForCausalLM.merge.__get__(model)
        return model
