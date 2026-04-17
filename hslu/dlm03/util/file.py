import ipywidgets
import jinja2
import pathlib
from pydantic import dataclasses

from hslu.dlm03.tools import unified_diff


@dataclasses.dataclass
class File:
    file_path: str
    content: str

    @classmethod
    def read_from(cls, path: pathlib.Path, *, root: pathlib.Path | None = None) -> "File":
        file_path = path.relative_to(root) if root is not None else path
        return cls(file_path=str(file_path), content=path.read_text())

    def diff(self, other: 'File', *, num_lines: int | None = None) -> unified_diff.UnifiedDiff:
        return unified_diff.diff(self.content, other.content, from_file=self.file_path, to_file=other.file_path,
                                 num_lines=num_lines)


@dataclasses.dataclass
class Files:
    files: list[File]

    @classmethod
    def read_from(cls, path: pathlib.Path, glob: str = "**/*") -> "Files":
        return cls(files=[File.read_from(file, root=path) for file in path.glob(glob) if file.is_file()])

    def diff(self, others: 'Files', *, num_lines: int | None = None) -> list[unified_diff.UnifiedDiff]:
        files_map = {file.file_path: file for file in self.files}
        others_map = {file.file_path: file for file in others.files}
        diffs = []
        for filename in files_map.keys() | others_map.keys():
            file = files_map.get(filename)
            other = others_map.get(filename)
            if other is None:
                other = file
            diffs.append(file.diff(other, num_lines=num_lines))
        return diffs
