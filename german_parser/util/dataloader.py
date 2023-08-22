import xml.etree.ElementTree as ET


class DatasetGenerator:
    def __init__(self, file_path: str):
        """initializes dataset generator

        Args:
            file_path (str): path to tiger.xml
        """

        with open(file_path, "rb") as f:
            self.document = ET.parse(f)

        self.all_terminals = self.document.findall(".//t")
        self.all_words = [t.attrib["word"] for t in self.all_terminals]