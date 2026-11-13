from toolkit.extension import Extension


class GODObjectiveEvaluatorExtension(Extension):
    uid = "god_objective_evaluator"
    name = "G.O.D objective evaluator"

    @classmethod
    def get_process(cls):
        from .objective_evaluator import GODObjectiveEvaluator

        return GODObjectiveEvaluator


AI_TOOLKIT_EXTENSIONS = [GODObjectiveEvaluatorExtension]
