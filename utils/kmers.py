import numpy as np
import pandas as pd
import itertools


class KMer_prot:
    def __init__(self, size: int = 3):
        """Divide dna or rna into substrings of size "size"
        Args:
            size (int, optional): size of each substring
        """
        self.k = size
        self.k_mers = None
        self.fite = False
        self.counts = 0

    def _get_all_combinations(self):
        """Gets the kmers
        Args:
            sequence (list): return a list within a list [["ACTGACATCTACT"]]
            IMPORTANT! USE [0] TO GET STRING!
        Returns:
            KMERs(np.array): Return a np.array with all kmers of all sequences [[kmer1],[kmer2],[kmer3]....]
        """
        alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "X",
            "B",
            "Z",
            "J",
        ]
        combinations = itertools.product(alphabet, repeat=self.k)
        return ("".join(tup) for tup in (combinations))

    def _get_kmers(self, sequence: str):
        """Helper fucntion to get Full combinations
        Uses itertoools to produce the product of size k
        """
        # Janela deslizante para obter os kmers de tamanho k
        return np.array(
            sequence[0][i : i + self.k] for i in range(len(sequence[0]) - self.k - 1)
        )

    def fit(self, sequences):
        """Does the fit
        Args:
            sequences (pd.DataFrame): Gets a dataset of nucleotides
        Returns:
            returns self
        """
        self.kmers = np.apply_along_axis(self._get_kmers, axis=1, arr=sequences)
        self.fite = True
        return self

    def _get_frq(self, seq):
        """Gets the counts for each count that appears in the sequence
        Args:
            seq (list): return a list within a list [["MGC...."]]
            IMPORTANT! USE [0] TO GET STRING!
        """
        combinations = self._get_all_combinations()
        counts = {kmer: 0 for kmer in combinations}
        for value in self.kmers[self.counts]:
            counts[value] += 1
        self.counts += 1
        return np.array([counts[i] / len(seq[0]) for i in counts])

    def transform(self, sequences: pd.DataFrame):
        """Transforms the dataset to get an informative value of the data
        Args:
            sequences (pd.DataFrame): amino acid dataset
        """
        if self.fite:
            freq = np.apply_along_axis(self._get_frq, axis=1, arr=sequences)
            return freq
        else:
            raise ("Please fit data first")

    def fit_transform(self, sequences: list):
        """Does fit and transform
        Args:
            sequences (pd.DataFrame): amino acid dataset
        Returns:
            _type_: Return the new dataset
        """
        sequences = pd.DataFrame(sequences)
        self.fit(sequences)
        new_data = self.transform(sequences)
        return new_data


if __name__ == "__main__":
    sequences = [
        "AAAAKAAALALLGEAPEVVDIWLPAGWRQPFRVFRLERKGDGVLV",
        "AAADGEPLHNEEERAGAGQVGRSLPQESEEQRTGSRPRRRRDLGSR",
        "AAAFSTPRATSYRILSSAGSGSTRADAPQVRRLHTTRDLLAKDYYA",
    ]
    kmer = KMer_prot()
    new_data = kmer.fit_transform(sequences)
    print(new_data)
