import json
import os
from pathlib import Path
from typing import Callable, List, Dict, Optional

try:
	from PIL import Image
except Exception:
	Image = None

try:
	import torch
	from torch.utils.data import Dataset as TorchDataset
	from torchvision import transforms as tv_transforms
except Exception:
	torch = None
	TorchDataset = object
	tv_transforms = None

try:
	import nltk
	from nltk.corpus import wordnet as wn
except Exception:
	wn = None


def build_dataset_json(
	mapping_txt: str,
	imagenet_root: str,
	out_json: str,
	use_wordnet: bool = False,
):
	"""
	Build a JSON dataset file from an ImageNet wnid->label mapping file.

	Each entry in the output JSON is a dict with keys:
	  - word: the primary label (first alias)
	  - definition: WordNet definition if available, otherwise the alias string
	  - synset_id: the ImageNet wnid (e.g. 'n01440764')
	  - image_dir: relative path to the folder containing validation images for that wnid

	If `use_wordnet` is True, NLTK's WordNet will be queried to populate `definition`.
	Make sure to run `nltk.download('wordnet')` locally before using that mode.
	"""
	entries: List[Dict] = []

	with open(mapping_txt, "r", encoding="utf-8") as f:
		lines = [l.strip() for l in f if l.strip()]

	for ln in lines:
		# mapping lines are of the form: wnid <space> name, alias, ...
		parts = ln.split(maxsplit=1)
		if len(parts) != 2:
			continue
		wnid, names = parts[0], parts[1]
		primary = names.split(",")[0].strip()
		definition = names

		if use_wordnet and wn is not None:
			try:
				# Convert wnid like 'n01440764' -> offset int(1440764)
				offset = int(wnid[1:])
				syn = wn.synset_from_pos_and_offset("n", offset)
				primary = syn.lemmas()[0].name()
				definition = syn.definition()
			except Exception:
				# fallback to mapping text
				pass

		image_dir = os.path.join(imagenet_root, wnid)
		entry = {
			"word": primary,
			"definition": definition,
			"synset_id": wnid,
			"image_dir": image_dir,
		}
		entries.append(entry)

	with open(out_json, "w", encoding="utf-8") as out:
		json.dump(entries, out, indent=2, ensure_ascii=False)

	return entries


class PseudoRDMIFDataset(TorchDataset):
	"""PyTorch Dataset for the pseudo RDMIF dataset.

	Expects a JSON file created by `build_dataset_json` where each item has at least
	`word`, `definition`, `synset_id`, and `image_dir`.

	On __getitem__ the dataset will pick an image file inside `image_dir` (deterministic
	selection by index modulo number of files in that dir) and return a dict:
	  {"word": str, "definition": str, "image": Tensor, "synset_id": str}
	"""

	IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG"}

	def __init__(
		self,
		json_path: str,
		transform: Optional[Callable] = None,
		loader: Callable = None,
	):
		with open(json_path, "r", encoding="utf-8") as f:
			self.items = json.load(f)
		if transform is not None:
			self.transform = transform
		else:
			if tv_transforms is not None:
				self.transform = tv_transforms.Compose([
					tv_transforms.Resize(256), tv_transforms.CenterCrop(224), tv_transforms.ToTensor()
				])
			else:
				self.transform = None
		self.loader = loader if loader is not None else self.default_loader

		# precompute per-item file lists (do not expand to every image to keep JSON small)
		for it in self.items:
			img_dir = Path(it.get("image_dir", ""))
			files = []
			if img_dir.exists() and img_dir.is_dir():
				for p in sorted(img_dir.iterdir()):
					if p.suffix in self.IMAGE_EXTS:
						files.append(str(p))
			it["_files"] = files

		# filter out any entries without images
		self.items = [it for it in self.items if it.get("_files")]

	def __len__(self):
		return len(self.items)

	def default_loader(self, path: str):
		return Image.open(path).convert("RGB")

	def __getitem__(self, index: int):
		item = self.items[index]
		files: List[str] = item.get("_files", [])
		if not files:
			raise RuntimeError(f"No images found for item {item}")

		# deterministic selection: choose by index modulo number of files
		file_path = files[index % len(files)]
		img = self.loader(file_path)
		if self.transform:
			img = self.transform(img)

		return {
			"word": item["word"],
			"definition": item["definition"],
			"synset_id": item.get("synset_id"),
			"image": img,
			"image_path": file_path,
		}

if __name__ == "__main__":
	# Convenience: create a dataset.json that uses the textual mapping as a fallback definition.
	repo_root = Path(__file__).parent
	mapping = repo_root / "data" / "LOC_synset_mappings.txt"
	imagenet_root = repo_root / "data" / "imagenet-val"
	out = repo_root / "dataset.json"

	if not out.exists():
		print("Building starter dataset.json (uses mapping text as definition).")
		build_dataset_json(str(mapping), str(imagenet_root), str(out), use_wordnet=False)
		print("Wrote", out)
	else:
		print(out, "already exists. To regenerate run build_dataset_json with use_wordnet=True after installing NLTK.")
