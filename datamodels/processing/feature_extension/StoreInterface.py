import os
import pickle as pkl


class StoreInterface:
    @classmethod
    def load_pkl(cls, path, filename):
        with open(os.path.join(path, filename),"rb") as f:
            [type, attrs] = pkl.load(f)
            obj = cls._get_subclass(type)()
            obj._set_attrs(**attrs)
            return obj

    @classmethod
    def _get_subclass(cls, type):
        if type in str(cls):
            return cls
        for subcl in cls._get_subclasses():
            if type in str(subcl):
                return subcl

    @classmethod
    def _get_subclasses(cls, list_subclasses=[]):
        for subclass in cls.__subclasses__():
            subclass._get_subclasses(list_subclasses)
            list_subclasses.append(subclass)
        return list_subclasses

    def _set_attrs(self, **kwargs):
        for name, val in kwargs.items():
            setattr(self, name, val)

    def _get_attrs(self):
        return self.__dict__

    def save_pkl(self, path, filename):
        attrs = self._get_attrs()
        with open(os.path.join(path, filename), "wb") as f:
            pkl.dump([self.__class__.__name__, attrs], f)