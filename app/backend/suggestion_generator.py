"""Gemini-based description suggestion generator."""

import json
from pathlib import Path
from google import genai


class SuggestionGenerator:
    def __init__(self, api_key: str, label_desc_path=None):
        self.label_descriptions = {}
        if label_desc_path:
            p = Path(label_desc_path)
            if p.exists():
                with open(p, "r") as f:
                    self.label_descriptions = json.load(f)

        prompt_path = Path(__file__).resolve().parent.parent / "data" / "suggestion_prompt.txt"
        self.prompt_template = prompt_path.read_text()

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def suggest(self, genre: str, label: str) -> str:
        """Generate a description suggestion for the given genre/label."""
        descs = self.label_descriptions.get(genre, {}).get(label, [])
        if descs:
            examples = "\n".join(f"- {d}" for d in descs[:5])
        else:
            examples = "(no reference descriptions available)"

        prompt = self.prompt_template.format(
            genre=genre, label=label, examples=examples,
        )

        response = self.client.models.generate_content(
            model=self.model, contents=prompt,
        )
        return response.text.strip()
