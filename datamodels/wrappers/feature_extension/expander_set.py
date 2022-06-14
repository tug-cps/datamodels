from typing import List
from .store_interface import StoreInterface
from sklearn.pipeline import Pipeline, make_pipeline
from .feature_expansion import FeatureExpansion
from .identity_expander import IdentityExpander


class ExpanderSet(StoreInterface):
    """
    This class implements a pipeline for feature expansion.
    """
    expander_pipeline: Pipeline = None

    def __init__(self, expanders: List[FeatureExpansion] = None, feature_names: List[str] = []):
        expanders = [IdentityExpander()] if expanders == None else expanders
        self.expander_pipeline = make_pipeline(*expanders, 'passthrough')
        self.expander_pipeline.steps[0][1].feature_names_in_ = feature_names

    @classmethod
    def from_names(cls, expander_names: List[str], feature_names: List[str]=[], **kwargs):
        """
        Create expander set from names
        """
        return cls([FeatureExpansion.from_name(name, **kwargs) for name in expander_names], feature_names)

    #################################### Getters and Setters ###########################################################

    def get_list_expanders(self):
        return [expander for (_, expander) in self.expander_pipeline.steps[:-1]]

    def get_expander_by_index(self, index=0):
        return self.expander_pipeline.steps[index][1] if index < len(self.expander_pipeline.steps) else None

    def set_feature_names(self, feature_names: List[str]=None):
        self.expander_pipeline.steps[0][1].feature_names_in_ = feature_names

    def get_feature_names(self, feature_names=None):
        return self.expander_pipeline.get_feature_names_out(feature_names)

    def type_exp_full(self):
        return "_".join(expander.__class__.__name__ for _, expander in self.expander_pipeline.steps)

    def type_last_exp(self):
        return self.expander_pipeline.steps[-2][1].__class__.__name__

    ########################################### Pipeline methods #######################################################

    def fit_transform(self, X, y=None, **fit_params):
        return self.expander_pipeline.fit_transform(X, y, **fit_params)

    def fit(self, X, y, **fit_params):
        self.expander_pipeline.fit(X, y, **fit_params)

    def transform(self, X):
        return self.expander_pipeline.transform(X)

    def get_num_output_feats(self):
        return self.get_list_expanders()[-1].get_num_output_feats()
