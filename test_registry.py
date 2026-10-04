import tempfile
import unittest
from pathlib import Path

from faceid.registry import FaceRegistry, cosine_similarity


class FaceRegistryTests(unittest.TestCase):
    def test_registered_embedding_survives_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "known_faces.json"
            FaceRegistry(registry_path).register("Ada", [1.0, 0.0, 0.0])

            reloaded = FaceRegistry(registry_path)

            match = reloaded.identify([0.99, 0.01, 0.0])
            self.assertIsNotNone(match)
            self.assertEqual(match[0], "Ada")

    def test_unknown_embedding_is_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = FaceRegistry(Path(directory) / "known_faces.json")
            registry.register("Ada", [1.0, 0.0])

            self.assertIsNone(registry.identify([0.0, 1.0]))

    def test_cosine_similarity_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()