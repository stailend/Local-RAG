import unittest

from local_rag.changes import change_digest


class FakeProvider:
    def __init__(self):
        self.prompts = []

    def summarize(self, text):
        self.prompts.append(text)
        return "The launch date changed from 14 to 15 February."


def document(text):
    return {"source": "/docs/project.md", "chunks": [{"locator": "file", "text": text}]}


class ChangesTest(unittest.TestCase):
    def test_reports_only_changed_content_to_model(self):
        provider = FakeProvider()
        old = document("The expedition starts on 14 February. The leader is Mira.")
        new = document("The expedition starts on 15 February. The leader is Mira.")
        result = change_digest(old, new, provider)
        self.assertIn("Updated", result)
        self.assertIn("-The expedition starts on 14 February.", provider.prompts[0])
        self.assertIn("+The expedition starts on 15 February.", provider.prompts[0])
        self.assertNotIn("leader is Mira", provider.prompts[0])

    def test_added_and_removed_files_are_summarized(self):
        provider = FakeProvider()
        self.assertTrue(change_digest(None, document("New finding."), provider).startswith("Added"))
        self.assertTrue(change_digest(document("Old finding."), None, provider).startswith("Removed"))


if __name__ == "__main__":
    unittest.main()
