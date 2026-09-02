import os
import sys
import unittest
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestDockerStack(unittest.TestCase):
    def test_compose_syntax_config(self):
        # Run docker compose config to validate compose syntax
        try:
            result = subprocess.run(
                ["docker", "compose", "config"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            # If we reach here, compose syntax is correct
            self.assertEqual(result.returncode, 0)
            self.assertIn("services", result.stdout)
            self.assertIn("collision-api", result.stdout)
            self.assertIn("collision-portal", result.stdout)
            self.assertIn("postgres", result.stdout)
        except subprocess.CalledProcessError as e:
            self.fail(f"docker compose config failed: {e.stderr}")
        except FileNotFoundError:
            self.skipTest("Docker Compose not found on system.")

    def test_dockerfiles_exist(self):
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "Dockerfile.api")))
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, "Dockerfile.portal")))

if __name__ == "__main__":
    unittest.main()
