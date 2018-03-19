import os
import os.path
import sys

sys.path.append("..")
from Utils.Singleton import Singleton
import cPickle
import yaml
import copy
import json
import h5py

FILE_MANAGER_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_MANAGER_FILENAME = "files.yaml"


class FilesManager(object):
    """
    Files Manager used to load and save any kind of files
    """
    __metaclass__ = Singleton

    def __init__(self, overrides_filename=None):
        """
        Constructor for FilesManager
        :param overrides_filename: "*.yaml file used to override paths to files
        """
        # save input data
        self.overrides_filename = overrides_filename
        # prints
        self.log = lambda x: self.log_str(x)

        # load file paths
        stream = file(os.path.join(FILE_MANAGER_PATH, FILE_MANAGER_FILENAME), 'r')
        self.files = yaml.load(stream)

        # override file paths
        if overrides_filename != None:
            print("FilesManager: overrides files according to " + str(overrides_filename))
            # read yaml file
            stream = file(os.path.join(FILE_MANAGER_PATH, overrides_filename), 'r')
            overrides = yaml.load(stream)
            # override
            self.override(self.files, overrides)

    def load_file(self, tokens, version=None):
        """
        load file given file tokens
        :param tokens: tokens delimited with '.' (each toked is a level in files.yaml)
        :param version: useful if old version is required (according to numbers in files.yaml)
        :return: data read from file according to file type
        """
        # get file path
        fileinfo = self.get_file_info(tokens, version)
        filetype = fileinfo["type"]
        filename = os.path.join(FILE_MANAGER_PATH, fileinfo["name"])

        # load data per file type
        if filetype == "pickle":
            self.log("FilesManager: Load pickle file: %s=%s" % (tokens, filename))
            picklefile = open(filename, "rb")

            # get number of objects stored in the pickle file
            nof_objects = 1
            if "nof_objects" in fileinfo:
                nof_objects = fileinfo["nof_objects"]

            # load data
            if nof_objects == 1:
                data = cPickle.load(picklefile)
            else:
                data = []
                for i in range(nof_objects):
                    data.append(cPickle.load(picklefile))

            picklefile.close()
            return data
        elif filetype == "yaml":
            self.log("FilesManager: Load yaml file: %s=%s" % (tokens, filename))
            stream = open(filename, 'r')
            data = yaml.load(stream)
            return data
        elif filetype == "json":
            self.log("FilesManager: Load json file: %s=%s" % (tokens, filename))
            f = open(filename, 'r')
            data = json.load(f)
            return data
        elif filetype == "h5py":
            self.log("FilesManager: Load h5py file: %s=%s" % (tokens, filename))
            data = h5py.File(filename, 'r')
            return data
        elif filetype == "text":
            self.log("FilesManager: Load text file: %s=%s" % (tokens, filename))
            with open(filename) as f:
                lines = f.readlines()
                return lines

    def save_file(self, tokens, data, version=None):
        """
        save file given tokens in pickle format
        :param tokens: tokens delimited with '.' (each toked is a level in files.yaml)
        :param version: useful if old version is required (according to numbers in files.yaml)
        :param data: data to save
        :return: void
        """
        # get file path
        fileinfo = self.get_file_info(tokens, version)
        filename = os.path.join(FILE_MANAGER_PATH, fileinfo["name"])

        self.log("FilesManager: Save pickle file: " + filename)

        # load data
        picklefile = open(filename, "wb")
        # get number of objects stored in the pickle file
        nof_objects = 1
        if "nof_objects" in fileinfo:
            nof_objects = fileinfo["nof_objects"]
        if nof_objects == 1:
            cPickle.dump(data, picklefile)
        else:
            for elem in data:
                cPickle.dump(elem, picklefile)

        picklefile.close()

    def file_exist(self, tokens, version=None):
        """
        check if file exists given tokens
        :param tokens: tokens delimited with '.' (each toked is a level in files.yaml)
        :param version: useful if old version is required (according to numbers in files.yaml)
        :return: True if file exist
        """
        # get file path
        fileinfo = self.get_file_info(tokens, version)
        filename = os.path.join(FILE_MANAGER_PATH, fileinfo["name"])

        return os.path.exists(filename)

    def get_file_info(self, tokens, version=None):

        """
        get file name given file tokens
        :param tokens: tokens delimited with '.' (each toked is a level in files.yaml)
        :param version: useful if old version is required (according to numbers in files.yaml)
        :return: dictionary with file info
        """
        # get list of tokens
        tokens_lst = tokens.split(".")

        # get filename
        fileinfo = self.files
        for token in tokens_lst:
            if fileinfo.has_key(token):
                fileinfo = fileinfo[token]
            else:
                raise Exception("unknown name token {0} for name {1}".format(token, tokens))

        # make sure fileinfo was extracted
        if not "name" in fileinfo:
            raise Exception("uncomplete file tokens", tokens)

        # handle versions
        if version is not None:
            if "versions" in fileinfo:
                versions = fileinfo["versions"]
                if version in versions:
                    # deep copy to be able to override info
                    fileinfo = copy.deepcopy(fileinfo)
                    fileinfo["name"] = versions[version]["name"]
                    self.log("FilesManager: %s  - Use Old Version %d" % (tokens, version))
                    if "doc" in  versions[version]:
                        self.log("FilesManager: Version %d doc - %s" % (version, versions[version]["doc"]))
                else:
                    raise Exception("FilesManager: %s version num %d wasn't found" % (tokens, version))
            else:
                raise Exception("FilesManager: %s versions token wasn't found" % (tokens))

        return fileinfo


    def get_file_path(self, tokens, version=None):
        """
        return full path to file
        :param tokens: tokens delimited with '.' (each toked is a level in files.yaml)
        :param version: useful if old version is required (according to numbers in files.yaml)
        :return: path to file
        """
        # get file path
        fileinfo = self.get_file_info(tokens, version)
        filename = os.path.join(FILE_MANAGER_PATH, fileinfo["name"])

        self.log("FilesManager: file path : %s=%s" % (tokens, filename))
        return filename

    def override(self, files_db, overrides):
        """
        Overrides files data base according to overrides data base
        :param files_db:
        :param overrides:
        :return: void
        """
        for elem in overrides:
            if elem in files_db:
                if type(overrides[elem]) == str:
                    files_db[elem] = overrides[elem]

                else:
                    self.override(files_db[elem], overrides[elem])
            else:
                files_db[elem] = overrides[elem]

    def add_logger(self, logger):
        """
        Log all prints to file
        :return:
        """
        self.log = lambda x: logger.log(x)

    def log_str(self, x):
        print(x)
