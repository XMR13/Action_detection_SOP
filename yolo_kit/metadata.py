from __future__ import annotations

from typing import Dict

def load_class_names(metadata_path: str) -> Dict[int, str]:
    """
    Load class names from the project's lightweight metadata YAML format (e.g. `configs/metadata.yaml`).

    The repo currently stores a simple mapping:
        names:
          0: person
          1: bicycle
          ...

    This function intentionally avoids adding a PyYAML dependency
    and use only pure dictionary based
    """

    names: Dict[int, str] = {}
    in_names = False

    #we first open the json metdata path
    with open(metadata_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            #if the metada starts with the notghing or comment
            if not line or line.startswith("#"):
                continue
            
            #check fot the names: metadata
            if line in ("names", "names:"):
                in_names= True
                continue
            if not in_names:
                continue

            # parse "id" label"
            if ":" not in line:
                continue
            
            #split into left and right for the id and the class
            left, right = line.split(":", 1)
            left = left.strip()
            right = right.strip().strip("'").strip('"')
            if not left.isdigit():
                continue
            names[int(left)] = right
    #return the names
    return names
