import os
import pickle as pkl


class StoreInterface:
    """
    Interface for storing objects as pickle file.
    """
    @classmethod
    def load_pkl(cls, path: str, filename: str):
        """
            Load from pickle file.
            @param path: directory containing file
            @param filename: filename
            @return: object from file
        """
        with open(os.path.join(path, filename),"rb") as f:
            [type, attrs] = pkl.load(f)
            obj = cls._get_type(type)()
            obj._set_attrs(**attrs)
            return obj

    def save_pkl(self, path, filename):
        """
            Save object to pickle file.
            @param path: directory containing file
            @param filename: filename
        """
        attrs = self._get_attrs()
        with open(os.path.join(path, filename), "wb") as f:
            pkl.dump([self.__class__.__name__, attrs], f)

    @classmethod
    def _get_type(cls, cls_type: str):
        """
        Get type of class from string (subclass of StoreInterface)
        Returns None if non-existent.
        @param cls_type: class name
        @return: class type
        """
        if cls_type in str(cls):
            return cls
        for subcl in cls._get_subclasses():
            if cls_type in str(subcl):
                return subcl
        return None

    @classmethod
    def _get_subclasses(cls, list_subclasses=[]):
        """
        Recursively get subclasses.
        @param list_subclasses: needed for recursion
        @return: list of subclass types
        """
        for subclass in cls.__subclasses__():
            subclass._get_subclasses(list_subclasses)
            list_subclasses.append(subclass)
        return list_subclasses

    def _set_attrs(self, **kwargs):
        """
        Set attributes of class - override if necessary
        """
        for name, val in kwargs.items():
            setattr(self, name, val)

    def _get_attrs(self):
        """
        Get attributes of class - override if necessary
        """
        return self.__dict__


