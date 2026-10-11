import os
import sqlite3
import tempfile
import unittest

from _support import engine, observe, simple

from stare_decisis import AuditLog, EventType, ManualClock
from stare_decisis.hashing import GENESIS_HASH


class LogTests(unittest.TestCase):
    def test_chain_links_and_verifies(self):
        log = AuditLog(clock=ManualClock())
        a = log.append(EventType.DECISION_LINKED, {"decision_id": "x", "path": []})
        b = log.append(EventType.DECISION_LINKED, {"decision_id": "y", "path": []})
        self.assertEqual(a.prev_hash, GENESIS_HASH)
        self.assertEqual(b.prev_hash, a.hash)
        self.assertEqual((a.seq, b.seq), (1, 2))
        self.assertTrue(log.verify().ok)

    def test_unknown_event_type_rejected(self):
        with self.assertRaises(ValueError):
            AuditLog().append("made_up", {})

    def test_update_and_delete_rejected(self):
        log = AuditLog()
        log.append(EventType.DECISION_LINKED, {"decision_id": "x", "path": []})
        with self.assertRaises(sqlite3.DatabaseError):
            log._conn.execute("UPDATE events SET payload = '{}' WHERE seq = 1")
        with self.assertRaises(sqlite3.DatabaseError):
            log._conn.execute("DELETE FROM events WHERE seq = 1")

    def test_tampering_detected_at_altered_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.db")
            log = AuditLog(path, clock=ManualClock())
            for i in range(5):
                log.append(EventType.DECISION_LINKED, {"decision_id": f"d{i}", "path": []})
            log.close()
            raw = sqlite3.connect(path)
            raw.execute("DROP TRIGGER events_no_update")
            raw.execute("""UPDATE events SET payload = '{"decision_id":"evil","path":[]}' WHERE seq = 3""")
            raw.commit()
            raw.close()
            check = AuditLog(path).verify()
            self.assertFalse(check.ok)
            self.assertEqual(check.first_bad_seq, 3)
            self.assertEqual(check.checked, 2)

    def test_valid_time_and_tx_time_are_distinct(self):
        eng = simple()
        node = observe(eng, "n1", 1, valid_time="2025-03-01")
        rec = eng.log.get(node.seq)
        self.assertEqual(rec.valid_time, "2025-03-01")
        self.assertTrue(rec.tx_time.startswith("2026-01-01"))

    def test_empty_log_passed_to_engine_is_used(self):
        # regression: an empty AuditLog is falsy (len 0) and was once replaced
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.db")
            eng = simple(path)
            observe(eng, "n1", 1)
            eng.run()
            self.assertIs(eng.log.path, path)
            self.assertGreater(len(AuditLog(path)), 0)

    def test_engine_with_default_log(self):
        eng = engine()
        self.assertEqual(len(eng.log), 0)


if __name__ == "__main__":
    unittest.main()
