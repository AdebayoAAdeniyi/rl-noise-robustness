"""Checks that the archived evidence in this repository is intact.

The figures are the record of runs that cannot be repeated cheaply, so their
integrity is worth asserting rather than assuming.
"""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ('Penalty-3x3-v0', 'DangerMaze-6x6-v0', 'CartPole-v1',
                'MountainCar-v0', 'CliffWalk-4x12-v0', 'CliffWalk-5x5-v0')
AFFECTED_METRICS = ('delta_error', 'reward_combined', 'average_return',
                    'performance_combined')


class TrainerTests(unittest.TestCase):
    def test_every_environment_has_its_trainer(self):
        for name in ('penalty.py', 'dangermaze.py', 'cliffwalk_4.py', 'cliffwalk_5.py'):
            self.assertTrue((ROOT / 'experiments/gridworld' / name).is_file(), name)
        for name in ('cartpole.py', 'mountaincar.py'):
            self.assertTrue((ROOT / 'experiments/classic_control' / name).is_file(), name)

    def test_no_machine_specific_paths_are_published(self):
        """Cluster home directories and personal working paths must not ship."""
        # Assembled at runtime so this file does not itself contain the strings.
        markers = ['/home/' + 'adebayo', '/media/' + 'adebayo',
                   '/home/' + 'azeez1', 'UofA' + 'Canada']
        for path in ROOT.rglob('*.py'):
            if path == Path(__file__):
                continue
            text = path.read_text(errors='ignore')
            for marker in markers:
                self.assertNotIn(marker, text, f'{path} still contains {marker}')


class FigureProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'results/ALL_FIGURE_PROVENANCE.json').read_text())

    def test_every_figure_matches_its_recorded_digest(self):
        for entry in self.manifest:
            path = ROOT / entry['repository_file']
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry['sha256'], path)
            self.assertEqual(len(data), entry['bytes'], path)

    def test_the_manifest_covers_every_figure_present(self):
        listed = {entry['repository_file'] for entry in self.manifest}
        present = {str(p.relative_to(ROOT)) for p in (ROOT / 'results/figures').rglob('*.png')}
        self.assertEqual(listed, present)

    def test_each_environment_has_its_gallery_and_readme(self):
        for env in ENVIRONMENTS:
            folder = ROOT / 'results/figures' / env
            self.assertTrue((folder / 'README.md').is_file(), env)
            self.assertEqual(len(list(folder.glob('*.png'))), 24, env)

    def test_affected_figures_are_declared_in_their_gallery_readme(self):
        """A figure built from a metric the logging defect touches must be named
        in its folder's README, so it is not read as sound. See docs/FINDINGS.md.
        """
        for env in ENVIRONMENTS:
            folder = ROOT / 'results/figures' / env
            readme = (folder / 'README.md').read_text()
            affected = [p.name for p in folder.glob('*.png')
                        if any(m in p.name for m in AFFECTED_METRICS)]
            self.assertEqual(len(affected), 8, env)
            for name in affected:
                self.assertIn(name, readme, f'{env}: {name} is not declared')


class DocumentationTests(unittest.TestCase):
    def test_the_documents_the_readme_points_at_exist(self):
        for name in ('FINDINGS.md', 'METHOD_NOTES.md', 'ENVIRONMENT.md'):
            self.assertTrue((ROOT / 'docs' / name).is_file(), name)

    def test_the_evidence_script_is_present(self):
        self.assertTrue((ROOT / 'verification/check_metric_logging.py').is_file())

    def test_the_report_is_included(self):
        self.assertTrue(
            (ROOT / 'docs/historical_reports/CMPUT655_project_report.pdf').is_file())


if __name__ == '__main__':
    unittest.main()
