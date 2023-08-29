import os
import re

class Cache:
    def __init__(self, folder, overwrite=False):
        self.folder = folder
        self.overwrite = overwrite

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def _cleanName(self, fileID):
        return re.sub(r"[^A-Za-z0-9-_]", "", fileID).lower()

    def _getFolderName(self, fileID):
        return os.path.join(self.folder, fileID[0:2])

    def _getFileName(self, fileID):
        fileID = self._cleanName(fileID)
        position = os.path.join(self.folder, fileID[0:2], fileID)
        return position

    def _createFolder(self, fileID):
        fileID = self._cleanName(fileID)
        subFolder = self._getFolderName(fileID)
        if not os.path.exists(subFolder):
            os.makedirs(subFolder)

    def getFile(self, fileID):
        position = self._getFileName(fileID)
        if os.path.exists(position):
            with open(position) as f:
                return f.read()
        return False

    def writeFile(self, fileID, content):
        position = self._getFileName(fileID)
        self._createFolder(fileID)
        if not os.path.exists(position) or self.overwrite:
            with open(position, "w") as fw:
                fw.write(content)
