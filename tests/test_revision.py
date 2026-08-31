"""Tests for revision.py — content-addressed revision store."""
import pytest
from fractions import Fraction
from martingale.revision import RevisionStore, Revision


class TestRevision:
    def _sample_policy_table(self):
        return {
            0: {0: Fraction(1, 3), 1: Fraction(2, 3)},
            1: {0: Fraction(1, 2), 1: Fraction(1, 2)},
        }

    def test_digest_is_deterministic(self):
        """Same policy table always produces same digest."""
        table = self._sample_policy_table()
        r1 = Revision(table)
        r2 = Revision(table)
        assert r1.digest == r2.digest

    def test_digest_is_hex_string(self):
        r = Revision(self._sample_policy_table())
        assert isinstance(r.digest, str)
        assert len(r.digest) == 64  # 32 bytes = 64 hex chars

    def test_different_tables_different_digests(self):
        t1 = {0: {0: Fraction(1, 3), 1: Fraction(2, 3)}}
        t2 = {0: {0: Fraction(2, 3), 1: Fraction(1, 3)}}
        r1 = Revision(t1)
        r2 = Revision(t2)
        assert r1.digest != r2.digest

    def test_canonical_serialization_is_stable(self):
        """Order of keys in the table doesn't change the digest."""
        t1 = {0: {0: Fraction(1, 3), 1: Fraction(2, 3)}}
        t2 = {0: {1: Fraction(2, 3), 0: Fraction(1, 3)}}  # different key order
        r1 = Revision(t1)
        r2 = Revision(t2)
        assert r1.digest == r2.digest


class TestRevisionStore:
    def _make_store(self, tmp_path):
        from martingale.revision import RevisionStore
        return RevisionStore(tmp_path)

    def test_store_and_retrieve(self, tmp_path):
        store = RevisionStore(tmp_path)
        table = {0: {0: Fraction(1, 2), 1: Fraction(1, 2)}}
        rev = store.publish(table)
        assert rev.digest in store
        retrieved = store.get(rev.digest)
        assert retrieved.table == rev.table

    def test_idempotent_publish(self, tmp_path):
        store = RevisionStore(tmp_path)
        table = {0: {0: Fraction(3, 4), 1: Fraction(1, 4)}}
        rev1 = store.publish(table)
        rev2 = store.publish(table)
        assert rev1.digest == rev2.digest
        assert len(list(store.all_digests())) == 1

    def test_missing_digest_raises(self, tmp_path):
        store = RevisionStore(tmp_path)
        with pytest.raises(KeyError):
            store.get("0" * 64)

    def test_multiple_revisions(self, tmp_path):
        store = RevisionStore(tmp_path)
        tables = [
            {0: {0: Fraction(i, 4), 1: Fraction(4 - i, 4)}}
            for i in range(1, 4)
        ]
        revisions = [store.publish(t) for t in tables]
        digests = {r.digest for r in revisions}
        assert len(digests) == 3
