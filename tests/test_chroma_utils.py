import unittest

from chroma_utils import batched_records


class BatchedRecordsTests(unittest.TestCase):
    def test_all_records_are_yielded_without_dropping_last_item(self) -> None:
        ids = [f"id-{index}" for index in range(5)]
        documents = [f"doc-{index}" for index in range(5)]
        metadatas = [{"index": index} for index in range(5)]

        batches = list(batched_records(ids, documents, metadatas, batch_size=2))

        flattened_ids = [item for batch_ids, _, _ in batches for item in batch_ids]
        self.assertEqual(flattened_ids, ids)
        self.assertEqual([len(batch_ids) for batch_ids, _, _ in batches], [2, 2, 1])

    def test_mismatched_lengths_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            list(batched_records(["id-1"], ["doc-1", "doc-2"], [{"index": 1}], batch_size=2))


if __name__ == "__main__":
    unittest.main()
