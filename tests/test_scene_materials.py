"""Profile-wide radio material defaults retain per-material override semantics."""
from types import SimpleNamespace
import unittest
from omegaconf import OmegaConf
from src.module_1_scene_gen.path_gen import apply_materials


class SceneMaterialTests(unittest.TestCase):
    def test_defaults_cover_unlisted_material_and_overrides_win(self):
        scene=SimpleNamespace(radio_materials={
            'itu_concrete':SimpleNamespace(scattering_coefficient=0.0),
            'unlisted_material':SimpleNamespace(scattering_coefficient=0.1)})
        cfg=OmegaConf.create({'material_config':{
            'defaults':{'scattering_coefficient':0.5},
            'overrides':{'concrete':{'scattering_coefficient':0.3}}}})
        apply_materials(scene,cfg)
        self.assertEqual(scene.radio_materials['itu_concrete'].scattering_coefficient,0.3)
        self.assertEqual(scene.radio_materials['unlisted_material'].scattering_coefficient,0.5)

    def test_legacy_profile_keeps_unlisted_material(self):
        scene=SimpleNamespace(radio_materials={
            'itu_concrete':SimpleNamespace(scattering_coefficient=0.0),
            'unlisted_material':SimpleNamespace(scattering_coefficient=0.1)})
        apply_materials(scene,OmegaConf.create({'material_config':{
            'overrides':{'concrete':{'scattering_coefficient':0.4}}}}))
        self.assertEqual(scene.radio_materials['unlisted_material'].scattering_coefficient,0.1)

if __name__=='__main__':
    unittest.main()
