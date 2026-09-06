"""URDF utilities for MIDAS retargeting-only frames."""

from __future__ import annotations

import atexit
import shutil
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

#: Fingertip frames, as ``parent link -> offset in that link's frame``. The
#: source URDF ends each digit at its DIP joint origin, but a Cartesian
#: objective needs the actual tip, so these are injected at load time.
#:
#: The thumb offset was ``0 -0.042 -0.010``, which placed thumb_tip 28 mm
#: *closer* to the palm than the thumb's own DIP joint — the "fingertip" sat
#: behind the knuckle, pointing back down the chain (cos -0.84 against the
#: thumb's distal direction). Any Cartesian retargeting then aimed the thumb at
#: a point inside the hand. Flipping the sign gives cos +0.96 and puts the tip
#: 41 mm distal, matching the fingers' 37 mm.
#:
#: These are CAD estimates and only the vector modes read them; the analytic
#: map never touches a tip frame, which is why this went unnoticed. Worth
#: measuring on the real hand.
TIP_LINKS = {
    "thumb_tip": ("thumb_dip", "0 0.042 -0.010"),
    "index_tip": ("index_dip_link", "0 0.036 -0.009"),
    "middle_tip": ("middle_dip_link", "0 0.036 -0.009"),
    "ring_tip": ("ring_dip_link", "0 0.036 -0.009"),
}


@lru_cache(maxsize=8)
def with_tip_links(urdf_path: str) -> str:
    """Return a URDF path with fixed fingertip frames added if missing.

    The source MIDAS URDF has distal link frames at the DIP joint origins. The
    vector retargeter needs actual fingertip frames so distal joints influence
    the Cartesian objective. This helper writes a small temporary URDF that
    references the original model content plus fixed, massless tip links.
    """

    source = Path(urdf_path).expanduser().resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    _rewrite_package_mesh_paths(root, source)
    existing_links = {link.attrib["name"] for link in root.findall("link")}

    added = False
    for tip_name, (parent_name, xyz) in TIP_LINKS.items():
        if tip_name in existing_links:
            continue
        ET.SubElement(root, "link", {"name": tip_name})
        joint = ET.SubElement(
            root,
            "joint",
            {"name": f"{tip_name}_fixed_joint", "type": "fixed"},
        )
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": parent_name})
        ET.SubElement(joint, "child", {"link": tip_name})
        added = True

    if not added:
        return str(source)

    temp_dir = Path(tempfile.mkdtemp(prefix="midas-hand-retargeter-"))
    # lru_cache bounds this per process, but without cleanup every run left a
    # directory behind in /tmp forever.
    atexit.register(shutil.rmtree, temp_dir, True)
    temp_path = temp_dir / source.name
    tree.write(temp_path, encoding="utf-8", xml_declaration=False)
    return str(temp_path)


def _rewrite_package_mesh_paths(root: ET.Element, source: Path) -> None:
    meshes_dir = source.parent.parent / "meshes"
    package_prefix = "package://midas_hand_urdf/meshes/"
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or not filename.startswith(package_prefix):
            continue
        mesh_name = filename.removeprefix(package_prefix)
        mesh.attrib["filename"] = str((meshes_dir / mesh_name).resolve())
