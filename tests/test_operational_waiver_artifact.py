import unittest
from pathlib import Path

import pandas as pd


class OperationalWaiverArtifactTests(unittest.TestCase):
    def test_waiver_profile_passes_operational_but_not_strict_gate(self):
        run = Path("runs/ews_full_20260813_operational_waiver")
        if not run.is_dir():
            self.skipTest("validated operational-waiver full run not present")
        gate = pd.read_csv(run / "deployment_gates.csv").iloc[0]
        self.assertTrue(gate["operational_core_gate"])
        self.assertTrue(gate["waiver_gate"])
        self.assertTrue(gate["operational_gate"])
        self.assertFalse(gate["strict_operational_gate"])
        self.assertFalse(gate["point_in_time_vintage_gate"])
        self.assertFalse(gate["investable_return_source_gate"])
        audit = pd.read_csv(run / "operational_risk_acceptance_audit.csv")
        self.assertEqual(len(audit), 3)
        self.assertTrue(audit["accepted"].all())
        self.assertTrue(audit["scope"].eq("operational_gate_only").all())


if __name__ == "__main__":
    unittest.main()
