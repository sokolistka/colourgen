"""Normalize and balance text-to-palette datasets.

Each row contributes one unit of mass, divided equally between its words.
Rows are then reweighted so every word receives approximately the same total
mass. The balanced training file is sampled with replacement; validation and
test files should be passed through unchanged.
"""

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix

from embedding import encode


TEXT_COLUMN = "text_input"
EMBEDDING_COLUMN = "text_embedding"
TOKEN_PATTERN = re.compile(r"[^\w']+", re.UNICODE)


def parse_tokens(value):
	"""Convert a CSV text-list value or plain phrase into normalized tokens."""
	try:
		parsed = ast.literal_eval(str(value))
	except (ValueError, SyntaxError):
		parsed = value

	if isinstance(parsed, (list, tuple)):
		text = " ".join(str(part) for part in parsed)
	else:
		text = str(parsed)

	tokens = [token.lower() for token in TOKEN_PATTERN.split(text) if token]
	return tokens or ["<empty>"]


def word_mass_weights(dataframe):
	"""Return row weights that balance total equal-per-word contribution."""
	token_rows = [parse_tokens(value) for value in dataframe[TEXT_COLUMN]]
	vocabulary = sorted({token for row in token_rows for token in row})
	token_index = {token: index for index, token in enumerate(vocabulary)}

	incidence = np.zeros((len(vocabulary), len(token_rows)), dtype=np.float64)
	for row_index, tokens in enumerate(token_rows):
		contribution = 1.0 / len(tokens)
		for token in set(tokens):
			incidence[token_index[token], row_index] = contribution

	# Solve for nonnegative row weights whose expected word mass is as equal
	# as the dataset structure allows. Sparse least squares avoids favoring
	# words merely because they occur in longer or more common phrases.
	sparse_incidence = csr_matrix(incidence)
	target_mass = sparse_incidence.sum() / len(vocabulary)
	solution = lsq_linear(
		sparse_incidence,
		np.full(len(vocabulary), target_mass),
		bounds=(1e-8, np.inf),
		lsmr_tol="auto",
		max_iter=1000
	)
	if not solution.success:
		raise RuntimeError(f"Could not balance dataset: {solution.message}")

	weights = solution.x
	weights /= weights.mean()

	return token_rows, vocabulary, weights, sparse_incidence


def normalize_and_balance(input_path, output_path, seed=42, balance=True):
	dataframe = pd.read_csv(input_path)
	if TEXT_COLUMN not in dataframe.columns:
		raise ValueError(f"Dataset must contain a {TEXT_COLUMN!r} column")

	token_rows, vocabulary, weights, incidence = word_mass_weights(dataframe)
	rng = np.random.default_rng(seed)
	if balance:
		selected = rng.choice(
			len(dataframe),
			size=len(dataframe),
			replace=True,
			p=weights / weights.sum()
		)
	else:
		selected = np.arange(len(dataframe))

	balanced = dataframe.iloc[selected].copy().reset_index(drop=True)
	normalized_texts = [
		" ".join(token_rows[index]) for index in selected
	]
	balanced[TEXT_COLUMN] = normalized_texts
	embeddings = encode(normalized_texts)
	balanced[EMBEDDING_COLUMN] = [
		json.dumps(vector.tolist(), separators=(",", ":"))
		for vector in embeddings
	]
	balanced.to_csv(output_path, index=False)

	expected_mass = incidence @ weights / weights.sum() * len(dataframe)
	mass_error = np.abs(expected_mass - expected_mass.mean())
	print(f"Input rows: {len(dataframe)}")
	print(f"Output rows: {len(balanced)}")
	print(f"Vocabulary size: {len(vocabulary)}")
	print(f"Expected word mass error (max): {mass_error.max():.6f}")
	print(f"Embedding column: {EMBEDDING_COLUMN} ({embeddings.shape[1]} values)")
	print(f"Output: {output_path}")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"input",
		type=Path,
		nargs="?",
		default=Path("palette_and_text_train.csv")
	)
	parser.add_argument(
		"output",
		type=Path,
		nargs="?",
		default=Path("palette_and_text_train_balanced.csv")
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--no-balance",
		action="store_true",
		help="Normalize and embed rows without resampling them"
	)
	args = parser.parse_args()
	normalize_and_balance(
		args.input,
		args.output,
		seed=args.seed,
		balance=not args.no_balance
	)


if __name__ == "__main__":
	main()
